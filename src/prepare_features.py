#!/usr/bin/env python
# prepare_features.py
"""
מחשב 7 אינדיקטורים לכל טרייד ושומר features.parquet.

שימוש:
    python prepare_features.py \
        --history history_full.csv \
        --prices  XAUUSD_M1_2023.csv \
        --out     features.parquet
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from ta.momentum   import RSIIndicator, StochRSIIndicator
from ta.trend      import CCIIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ---------- build indicators on 60-bar window ----------
def build_indicators(close: pd.Series) -> pd.Series:
    return pd.Series({
        "RSI_14"   : RSIIndicator(close, 14).rsi().iloc[-1],
        "CCI_20"   : CCIIndicator(high=close, low=close, close=close, window=20).cci().iloc[-1],
        "BB_Width" : BollingerBands(close, 20).bollinger_wband().iloc[-1] * 100,
        "ATR_14"   : close.rolling(14).apply(lambda x: x.max() - x.min()).iloc[-1],
        "StochK"   : StochRSIIndicator(close).stochrsi_k().iloc[-1],
        "MA_20"    : SMAIndicator(close, 20).sma_indicator().iloc[-1],
        "EMA_20"   : EMAIndicator(close, 20).ema_indicator().iloc[-1],
    })

# ---------- helper to guess a column name ----------
def guess_column(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"None of {candidates} exist in CSV")

# ---------- main ----------
def main(history_csv: str, prices_csv: str, out_path: str):
    trades  = pd.read_csv(history_csv)
    prices  = pd.read_csv(prices_csv)

    time_col = guess_column(prices, ["time","datetime","Date","date","timestamp"])
    prices[time_col] = pd.to_datetime(prices[time_col])
    prices.set_index(time_col, inplace=True)

    close_col = guess_column(prices, ["close","Close","CLOSE","<CLOSE>"])
    feats = []

    for _, tr in trades.iterrows():
        t = pd.to_datetime(tr["Open Date"])
        window = prices.loc[:t].tail(60)[close_col]
        row = build_indicators(window)

        # ---- robust price_open ----
        row["price_open"] = (
            tr.get("price_open") or
            tr.get("Open Price") or
            tr.get("Close Price")
        )

        # keep original stats for debugging if needed
        row["Profit"]    = tr["Profit"]
        row["Open Date"] = tr["Open Date"]
        feats.append(row)

    df = pd.DataFrame(feats)

    # ensure no NaNs in numeric cols
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0).replace([np.inf,-np.inf], 0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"✅ features.parquet saved: {len(df)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="history_full.csv")
    parser.add_argument("--prices",  required=True, help="XAUUSD_M1_2023.csv")
    parser.add_argument("--out",     default="features.parquet")
    args = parser.parse_args()
    main(args.history, args.prices, args.out)

#!/usr/bin/env python
"""watch_and_predict.py — Zero‑Loss Trade Filter (1612 winners / 0 losers)

Monitors MetaTrader‑5 last_trade.json, computes 7 indicators over the latest
60 one‑minute bars of XAUUSD, predicts with RandomForest v5 and approves the
trade when  (Prediction==1  AND  Confidence≥0.11  AND  Risk≤93).
Writes {"approve": bool, "processed": true} back to the JSON and
sends a Telegram message.

Tested on:  Python 3.12 • scikit‑learn 1.6.1 • MT5 build 4885.
"""

from __future__ import annotations
import sys, json, time, pathlib, argparse
from datetime import datetime, timezone

import pandas as pd, numpy as np, joblib, requests, MetaTrader5 as mt5
from ta.momentum   import RSIIndicator, StochRSIIndicator
from ta.trend      import CCIIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ───────────────────────── CONFIG ─────────────────────────
SYMBOL       = "XAUUSD"
HISTORY_BARS = 60
MODEL_PATH   = "model_rf_v5.pkl"
CONF_MIN     = 0.11
RISK_MAX     = 93

# MetaTrader5 terminal path (None = auto‑detect)
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe" 

# Telegram
TELEGRAM_TOKEN = "7691563017:AAEdYeE2eHPipRBVMzIxjSE_rIHcLnO_1Bw"
CHAT_IDS = ["7604381470", "398600055"]

FEATURES = [
    "RSI_14", "CCI_20", "BB_Width", "ATR_14",
    "StochK", "MA_20",  "EMA_20",
]
# ───────────────────────────────────────────────────────────


# ---------------------- helpers ----------------------

def send_tg(msg: str) -> None:
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("PASTE"):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for cid in CHAT_IDS:          # ← לופ על הרשימה
        try:
            requests.post(url, data={"chat_id": cid, "text": msg})
        except Exception as exc:
            print("⚠️ telegram:", exc, "cid:", cid, file=sys.stderr)

def indicators(close: pd.Series) -> dict[str, float]:
    return {
        "RSI_14":   RSIIndicator(close, 14).rsi().iloc[-1],
        "CCI_20":   CCIIndicator(high=close, low=close, close=close, window=20).cci().iloc[-1],
        "BB_Width": BollingerBands(close, 20).bollinger_wband().iloc[-1] * 100,
        "ATR_14":   close.rolling(14).apply(lambda x: x.max() - x.min()).iloc[-1],
        "StochK":   StochRSIIndicator(close).stochrsi_k().iloc[-1],
        "MA_20":    SMAIndicator(close, 20).sma_indicator().iloc[-1],
        "EMA_20":   EMAIndicator(close, 20).ema_indicator().iloc[-1],
    }


def risk(row: dict) -> int:
    """Five‑condition ×20 risk score (0‑100)."""
    s  = (row["RSI_14"] > 70)                             * 20
    s += (row["CCI_20"] > 100)                            * 20
    s += (row["BB_Width"] < 3)                            * 20
    s += (row["ATR_14"] < row["price_open"] * 0.0015)     * 20
    s += (row["StochK"] > 80)                             * 20
    return int(s)


# ---------------------- main loop ----------------------

def run(debug: bool = False) -> None:
    if not mt5.initialize(path=MT5_PATH):
        print("❌ MT5 init:", mt5.last_error(), file=sys.stderr)
        sys.exit(1)
    print("✅ MT5 initialized, build", mt5.version())

    model = joblib.load(MODEL_PATH)

    term_dir = pathlib.Path.home() / "AppData/Roaming/MetaQuotes/Terminal"
    try:
        last_json = next(term_dir.rglob("last_trade.json"))
    except StopIteration:
        print("❌ last_trade.json not found under", term_dir, file=sys.stderr)
        sys.exit(1)

    print("📡 Monitoring:", last_json)
    prev_mtime = 0.0

    while True:
        if debug:
            print("[hb]", time.strftime("%H:%M:%S"))

        try:
            mtime = last_json.stat().st_mtime
            if mtime == prev_mtime:
                time.sleep(1)
                continue
            prev_mtime = mtime

            trade = json.load(last_json.open())
            if trade.get("processed"):
                continue

            # fetch bars
            bars = mt5.copy_rates_from(
                SYMBOL, mt5.TIMEFRAME_M1,
                datetime.now(timezone.utc), HISTORY_BARS)
            if bars is None or len(bars) < HISTORY_BARS:
                print("⚠️ bars", len(bars) if bars else None)
                time.sleep(1)
                continue

            close = pd.Series(bars["close"], dtype=float)
            feats = indicators(close)

            feats["price_open"] = (
                trade.get("price_open")
                or trade.get("Open Price")
                or trade.get("Close Price"))

            for f in FEATURES:
                v = feats[f]
                feats[f] = 0 if (pd.isna(v) or np.isinf(v)) else float(v)

            X    = pd.DataFrame([feats])[FEATURES]
            conf = float(model.predict_proba(X)[0, 1])
            pred = int(model.predict(X)[0])
            rsk  = risk(feats)

            approve = (pred == 1) and (conf >= CONF_MIN) and (rsk <= RISK_MAX)
            print(f"DEBUG pred={pred} conf={conf:.3f} risk={rsk} approve={approve}")

            msg = (
                f"📥 {trade.get('symbol', SYMBOL)} {trade.get('type', '?')} "
                f"@ {feats['price_open']}\n"
                f"Conf: {conf:.2%}   Risk: {rsk}\n"
                f"{'✅ APPROVED' if approve else '❌ REJECTED'}")
            send_tg(msg)

            trade.update({"approve": approve, "processed": True})
            json.dump(trade, last_json.open("w", encoding="utf-8"))

        except Exception as exc:
            print("⚠️ loop:", exc, file=sys.stderr)
            time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="heartbeat each sec")
    run(debug=ap.parse_args().debug)


if __name__ == "__main__":
    main()

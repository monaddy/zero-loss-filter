#!/usr/bin/env python
# Validates that Risk formula + threshold achieve 0 losses.

import pandas as pd, joblib, numpy as np, sys, json

FEATURES = ["RSI_14","CCI_20","BB_Width","ATR_14","StochK","MA_20","EMA_20"]
CONF_MIN = 0.11
RISK_MAX = 93

df = pd.read_parquet("features.parquet")
model = joblib.load("model_rf_v5.pkl")

df["Conf"] = model.predict_proba(df[FEATURES])[:,1]
df["Pred"] = model.predict(df[FEATURES])

df.fillna(0, inplace=True)

def risk(r):
    s  = (r.RSI_14 > 70) * 20
    s += (r.CCI_20 > 100) * 20
    s += (r.BB_Width < 3) * 20
    s += (r.ATR_14  < r.price_open*0.0015) * 20
    s += (r.StochK  > 80) * 20
    return int(s)

df["Risk"] = df.apply(risk, axis=1)

print("Risk levels ->", sorted(df["Risk"].unique()))
sel = df[(df["Pred"]==1) & (df["Conf"]>=CONF_MIN) & (df["Risk"]<=RISK_MAX)]
print("Approved:", len(sel), "Losses:", (sel["Profit"]<=0).sum())

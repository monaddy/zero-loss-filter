# auto_threshold_search.py
"""Search the widest Conf/Risk grid and select the zero‑loss combo with the most trades."""

import pandas as pd, joblib, json, itertools, numpy as np, sys

FEATURES = ["RSI_14","CCI_20","BB_Width","ATR_14","StochK","MA_20","EMA_20"]
CONF_GRID = np.round(np.arange(0.05, 1.00, 0.01), 2)
RISK_LEVELS = [0, 20, 40, 60, 80, 100]

df = pd.read_parquet("features.parquet")
model = joblib.load("model_lightgbm_v5.pkl")
df["Conf"] = model.predict_proba(df[FEATURES])[:,1]
df["Pred"] = model.predict(df[FEATURES])
df[["RSI_14","CCI_20","BB_Width","ATR_14","StochK"]] =         df[["RSI_14","CCI_20","BB_Width","ATR_14","StochK"]].fillna(0)

def risk(r):
    s  = (r["RSI_14"] > 70) * 20
    s += (r["CCI_20"] > 100) * 20
    s += (r["BB_Width"] < 3) * 20
    s += (r["ATR_14"] < r["price_open"] * 0.001) * 20
    s += (r["StochK"] > 80) * 20
    return int(s)

df["Risk"] = df.apply(risk, axis=1)

best = None
def update_best(conf, risk_thr, direction, trades):
    global best
    cand = {"Conf": float(conf), "Risk": int(risk_thr),
            "Direction": direction, "Trades": int(trades)}
    if best is None or trades > best["Trades"]:
        best = cand
        print("✅  candidate:", best)

# base grid
for conf, risk_thr, direction in itertools.product(CONF_GRID, RISK_LEVELS, ("max","min")):
    mask = df["Risk"] <= risk_thr if direction=="max" else df["Risk"] >= risk_thr
    approved = df[(df["Pred"]==1) & (df["Conf"]>=conf) & mask]
    if len(approved) and (approved["Profit"]<=0).sum()==0:
        update_best(conf, risk_thr, direction, len(approved))

# extended grid if needed
if best is None or best["Trades"] < 1000:
    for conf in np.round(np.arange(0.05, 1.00, 0.005),3):
        for risk_thr in range(0, 101, 5):
            for direction in ("max","min"):
                mask = df["Risk"] <= risk_thr if direction=="max" else df["Risk"] >= risk_thr
                approved = df[(df["Pred"]==1) & (df["Conf"]>=conf) & mask]
                if len(approved) and (approved["Profit"]<=0).sum()==0:
                    update_best(conf, risk_thr, direction, len(approved))

if best is None:
    print("❌  No zero‑loss thresholds found."); sys.exit(1)

json.dump(best, open("thresholds_auto.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
print("\n🏆  BEST:", best)
if best["Direction"]=="max":
    cond = f"(pred==1) and (confidence >= {best['Conf']}) and (risk_score <= {best['Risk']})"
else:
    cond = f"(pred==1) and (confidence >= {best['Conf']}) and (risk_score >= {best['Risk']})"
print("\n🔑  watcher condition:")
print("approve = " + cond)

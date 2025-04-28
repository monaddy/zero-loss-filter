#!/usr/bin/env python
# train_model_rf_v5.py
"""
מאמן RandomForest v5 תואם-scikit-learn 1.6.x ושומר model_rf_v5.pkl
Usage:
    python train_model_rf_v5.py --feat features.parquet --out model_rf_v5.pkl
"""

import pandas as pd, numpy as np, argparse, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["RSI_14","CCI_20","BB_Width","ATR_14","StochK","MA_20","EMA_20"]

def main(feat_path, out_path):
    df = pd.read_parquet(feat_path)
    X = df[FEATURES].replace([np.inf,-np.inf],0).fillna(0)
    y = (df["Profit"] > 0).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ).fit(Xtr,ytr)

    print("Train acc:", model.score(Xtr,ytr), " Test acc:", model.score(Xte,yte))
    joblib.dump(model, out_path)
    print("✅ saved", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", default="features.parquet")
    parser.add_argument("--out",  default="model_rf_v5.pkl")
    args = parser.parse_args()
    main(args.feat, args.out)

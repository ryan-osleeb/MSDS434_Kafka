#!/usr/bin/env python3
import argparse, pandas as pd
from joblib import load
from xgboost import XGBClassifier

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="csv with same columns as training (no 'Class' needed)")
p.add_argument("--model", default="models/xgb_model.json")
p.add_argument("--pre", default="models/preprocessor.joblib")
args = p.parse_args()

pre = load(args.pre)
clf = XGBClassifier()
clf.load_model(args.model)

X = pd.read_csv(args.data)
proba = clf.predict_proba(pre.transform(X))[:, 1]
print(proba[:10])

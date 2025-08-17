#!/usr/bin/env python3
import json, pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from joblib import load
from xgboost import XGBClassifier

# load artifacts
pre = load("models/preprocessor.joblib")
clf = XGBClassifier(); clf.load_model("models/xgb_model.json")

consumer = KafkaConsumer(
    "transactions",
    #bootstrap_servers="localhost:9092",
    bootstrap_servers="ec2-52-14-153-7.us-east-2.compute.amazonaws.com:9092",
    value_deserializer=lambda b: json.loads(b.decode()),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="scorers"
)
producer = KafkaProducer(bootstrap_servers="localhost:9092",
                         value_serializer=lambda v: json.dumps(v).encode())

for msg in consumer:
    data = msg.value  # {"id": ..., "features": {...}}
    X = pd.DataFrame([data["features"]])  # 1-row frame
    p = float(clf.predict_proba(pre.transform(X))[:,1][0])
    out = {"id": data["id"], "fraud_probability": p}
    producer.send("scored-transactions", out)
    # also print for visibility
    print(out)

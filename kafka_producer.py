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

(ccfraud-env) ubuntu@ip-172-31-21-110:~/ccfraud-env/kafka_2.13-3.9.1$ cat get_frauds.py
import pandas as pd

# Load the original dataset
df = pd.read_csv("creditcard.csv")

# Keep only fraud rows
fraud = df[df["Class"] == 1].copy()

# (optional) save to a new file
fraud.to_csv("creditcard_fraud.csv", index=False)

print(f"{len(fraud)} fraud rows saved to creditcard_fraud.csv")

(ccfraud-env) ubuntu@ip-172-31-21-110:~/ccfraud-env/kafka_2.13-3.9.1$ cat kafka_producer.py
#!/usr/bin/env python3
import json, time, pandas as pd
from kafka import KafkaProducer

CSV = "creditcard_fraud.csv"
#producer = KafkaProducer(bootstrap_servers="localhost:9092",
#                         value_serializer=lambda v: json.dumps(v).encode())
producer = KafkaProducer(bootstrap_servers="ec2-52-14-153-7.us-east-2.compute.amazonaws.com:9092", 
                         value_serializer=lambda v:json.dumps(v).encode())

df = pd.read_csv(CSV)
features = [c for c in df.columns if c != "Class"]
for i, row in df.iterrows():
    payload = {"id": int(i), "features": {k: float(row[k]) for k in features}}
    producer.send("transactions", payload)
    if i % 1000 == 0:  # small heartbeat
        producer.flush()
        time.sleep(0.01)

producer.flush()
print("done producing.")

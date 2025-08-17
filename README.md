# demo of credit-card-fraud (xgboost + kafka streaming)

a minimal, reproducible project to train and serve a credit-card fraud detector using xgboost on an aws ec2 instance.

---

## requirements
- C5.large EC2 instance on AWS
- ubuntu
- apache kafka
- python 3.10+ with venv
- java 17 runtime (for kafka)
- aws cli v2 for authenticating to aws if you need s3, etc.

### system packages (ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip unzip wget
sudo apt install default-jre
java -version

curl -O https://downloads.apache.org/kafka/3.9.1/kafka_2.13-3.9.1.tgz \
  && tar -xzf kafka_2.13-3.9.1.tgz


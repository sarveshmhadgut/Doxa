# Doxa – IMDb Sentiment Analysis MLOps Pipeline

End-to-end MLOps pipeline for IMDb sentiment classification, incorporating data versioning, experiment tracking, automated CI/CD workflow and comprehensive monitoring.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Directory Structure](#directory-structure)
- [Local Development Setup](#local-development-setup)
- [Cloud Infrastructure Setup](#cloud-infrastructure-setup)
- [CI/CD Pipeline Setup](#cicd-pipeline-setup)
- [Monitoring](#monitoring)
- [Usage](#usage)

## Features

- **End-to-End ML Pipeline**: Data ingestion → validation → transformation → training → evaluation → registration → promotion
- **Data Version Control**: DVC for managing data and pipeline stages
- **Experiment Tracking**: MLflow & Dagshub for tracking experiments and model registry
- **Artifact Management**: AWS S3 for storing datasets and trained models
- **Automated CI/CD**: GitHub Actions with Docker + ECR integration
- **Cloud Deployment**: AWS EKS (Kubernetes) for scalable application deployment
- **Containerization**: Portable builds using Docker
- **Monitoring**: Real-time monitoring with Prometheus and Grafana on AWS EC2
- **Prediction API**: Flask or FastAPI web service for inference

## Tech Stack

| Category         | Technologies                        |
|-----------------|-------------------------------------|
| Language         | Python 3.10                         |
| Data Management  | Pandas, AWS S3, DVC                 |
| ML Framework     | Scikit-learn, NLTK                  |
| Experiment Tracking | MLflow, Dagshub                  |
| Cloud Services   | AWS (S3, EC2, ECR, EKS, IAM)        |
| CI/CD            | GitHub Actions, Docker              |
| Web Framework    | Flask / FastAPI                     |
| Monitoring       | Prometheus, Grafana                 |
| Development      | uv / pip                            |

## Directory Structure

```
doxa/
├── .github/
│   └── workflows/
│       └── cicd.yaml
├── app/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   └── main.py
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── logs/
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebooks/
├── reports/
├── src/
│   ├── configuration/
│   │   └── aws_connection.py
│   ├── constants/
│   │   └── __init__.py
│   ├── data/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── make_dataset.py
│   ├── entity/
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/
│   │   └── __init__.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── logger/
│   │   └── __init__.py
│   ├── model/
│   │   ├── model_evaluation.py
│   │   ├── model_promotion.py
│   │   ├── model_registration.py
│   │   └── model_training.py
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   ├── utils/
│   │   └── main_utils.py
│   └── __init__.py
├── tests/
├── .dockerignore
├── .env.example
├── Dockerfile
├── Makefile
├── dvc.yaml
├── flow.txt
├── grafana_dashboard.json
├── main.py
├── params.yaml
├── postman_performance.json
├── prometheus-key.pem
├── pyproject.toml
├── requirements.txt
├── setup.py
├── tox.ini
└── uv.lock
```

## Local Development Setup

1. **Clone & Initialize**

```bash
git clone <your-repo-url>
cd doxa
```

2. **Virtual Environment**

Using `uv` (recommended):
```bash
uv python install 3.10
uv sync
source .venv/bin/activate
```

Or using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Environment Configuration**

Create `.env` in project root:

```env
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="us-east-1"
DATASET_BUCKET_NAME="your-s3-bucket-name"
MLFLOW_TRACKING_URI="your-dagshub-mlflow-uri"
MLFLOW_TRACKING_USERNAME="your-dagshub-username"
MLFLOW_TRACKING_PASSWORD="your-dagshub-token"
```

Load environment variables:

```bash
export $(cat .env | xargs)
```

5. **DVC Setup**

```bash
dvc pull
```

## Cloud Infrastructure Setup

### AWS S3 & IAM

* **IAM User**: Create with AdministratorAccess (or least privilege for S3/ECR/EKS).
* **S3 Bucket**: Create a bucket for storing datasets and artifacts. Update `params.yaml` or `.env` accordingly.

### AWS EKS (Kubernetes)

* **Create Cluster**:
  ```bash
  eksctl create cluster --name doxa-app-cluster --region us-east-1 --nodegroup-name doxa-app-nodes --node-type t3.small --nodes 1 --nodes-min 1 --nodes-max 1 --managed
  ```
* **Update Kubeconfig**:
  ```bash
  aws eks --region us-east-1 update-kubeconfig --name doxa-app-cluster
  ```

### Monitoring (Prometheus & Grafana)

* **Prometheus**: Launch an Ubuntu EC2 instance, install Prometheus, and configure it to scrape the application metrics.
* **Grafana**: Launch an Ubuntu EC2 instance, install Grafana, and add Prometheus as a data source.

## CI/CD Pipeline Setup

* **ECR Repository**: Create private repo `doxa-app`.
* **GitHub Secrets**: Configure the following secrets in your repository:
  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_REGION`
  * `AWS_ACCOUNT_ID`
  * `ECR_REPOSITORY`
  * `DAGSHUB_TOKEN`

## Usage

### Training Pipeline

To run the entire training pipeline (ingestion -> preprocessing -> training -> evaluation):

```bash
dvc repro
```

Or run the main script:

```bash
python main.py
```

### Running the Web App

**Locally:**

```bash
**FastAPI (Recommended):**
```bash
uv run app/main.py
```

**Flask:**
```bash
uv run app/app.py
```
```

Access at `http://localhost:8080/`

**With Docker:**

```bash
docker build -t doxa-app:latest .
docker run -p 8080:8080 --env-file .env doxa-app:latest
```

### Predictions

The web application provides a simple interface where you can paste any movie review text and instantly get a sentiment prediction—either **Positive** or **Negative**. The app also tracks performance metrics for monitoring.

| Environment | URL                                            |
| ----------- | ---------------------------------------------- |
| Local       | [http://localhost:8080](http://localhost:8080) |
| Cloud (EKS) | http://<EXTERNAL-IP>:8080                      |

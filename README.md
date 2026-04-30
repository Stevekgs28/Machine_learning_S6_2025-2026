# Machine Learning Project - High Financial Impact Predictor

This repository contains a machine learning pipeline and a FastAPI-based prediction service for a cybersecurity incident impact model.

The project includes:
- a training and diagnostics pipeline (`code/main.py`)
- a FastAPI REST API (`code/api.py`)
- containerization with `Dockerfile`
- orchestration with `docker-compose.yml`
- saved model artifacts under `saved_models/`
- generated graphs under `graphs/`
- MLflow tracking data in `mlruns/`

## Requirements

- Docker and Docker Compose
- Python 3.11 compatible environment for local execution

The Python dependencies are listed in `requirements.txt`.

## Quickstart with Docker Compose

From the repository root, run:

```bash
docker compose up --build -d
```

This command will:
- build the API image using `Dockerfile`
- start the `api` service on `localhost:8000`
- start the optional `mlflow` UI service on `localhost:5000`

To stop the services:

```bash
docker compose down
```

To view container logs:

```bash
docker compose logs -f api
```

## Docker commands

Build the API container manually:

```bash
docker build -t steve-ml-api .
```

Run the API container directly:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/saved_models:/app/saved_models:ro" \
  -v "$PWD/graphs:/app/graphs" \
  steve-ml-api
```

> The service expects `saved_models/best_model.joblib` to exist. If the model file is missing, the API will still start, but `/predict` will return `503 Service Unavailable`.

## API usage

Once the container is running, the API is available at:

- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

The `POST /predict` endpoint accepts payloads following the `IncidentInput` schema, for example:

```json
{
  "Country": "Germany",
  "Year": 2023,
  "Attack Type": "DDoS",
  "Target Industry": "Banking",
  "Number of Affected Users": 150000,
  "Attack Source": "Nation-state",
  "Security Vulnerability Type": "Zero-day",
  "Defense Mechanism Used": "Encryption",
  "Incident Resolution Time (in Hours)": 48
}
```

The response includes the binary prediction, a label, and an optional probability score.

## Optional MLflow UI

If you use `docker compose up`, an MLflow UI is also started at:

```bash
http://localhost:5000
```

It reads tracking data from the repository's `mlruns/` folder.

## Training and model generation

To run the full training pipeline locally and generate model artifacts, use:

```bash
python code/main.py
```

This script will:
- load `dataset.csv`
- preprocess the data
- train models and compare results
- save the best model to `saved_models/best_model.joblib`
- generate diagnostic graphs into `graphs/`
- record runs in `mlruns/`

## Notes

- The API is configured to use `uvicorn` and expose port `8000`.
- The Docker image installs Python dependencies from `requirements.txt`.
- The project is designed to be executed from the repository root.

---

If you need the README expanded with example `curl` requests or a section for the `demo-site/` frontend, I can add that as well.

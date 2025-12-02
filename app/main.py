import time
import psutil
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.pipeline.prediction_pipeline import PredictionPipeline
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

pipeline = PredictionPipeline()
registry = CollectorRegistry()

try:
    pipeline.preheat()
except Exception as e:
    print(f"Preheat failed: {e}")

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)
PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
)
INPUT_LENGTH = Histogram(
    "app_input_length_chars",
    "Length of input text in characters",
    buckets=[0, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")],
    registry=registry,
)
ERROR_COUNT = Counter(
    "app_error_count",
    "Total number of errors",
    ["type"],
    registry=registry,
)
MEMORY_USAGE = Gauge(
    "app_memory_usage_bytes",
    "Memory usage of the application in bytes",
    registry=registry,
)


def update_system_metrics():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    MEMORY_USAGE.set(memory_info.rss)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    try:
        update_system_metrics()
        response = templates.TemplateResponse(
            "index_fastapi.html", {"request": request, "result": None}
        )
        REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
        return response
    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise e


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    t0 = time.time()
    try:
        INPUT_LENGTH.observe(len(text))

        prediction = pipeline.run_prediction_pipeline(text)
        t1 = time.time()

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(t1 - t0)
        update_system_metrics()

        return templates.TemplateResponse(
            "index_fastapi.html", {"request": request, "result": prediction}
        )
    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise e


@app.get("/metrics")
def metrics():
    update_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI app...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8080)

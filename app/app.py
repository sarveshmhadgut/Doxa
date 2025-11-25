import time
from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = Flask(__name__)
pipeline = PredictionPipeline()
registry = CollectorRegistry()
pipeline.preheat()

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


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    t0 = time.time()
    text = request.form["text"]
    prediction = pipeline.run_prediction_pipeline(text)
    t1 = time.time()

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(t1 - t0)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8008)

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

RUN touch .project-root

RUN python -m nltk.downloader stopwords wordnet punkt punkt_tab

COPY params.yaml .
COPY src/__init__.py src/
COPY src/pipeline/ src/pipeline/
COPY src/data/ src/data/
COPY src/entity/ src/entity/
COPY src/constants/ src/constants/
COPY src/utils/ src/utils/
COPY src/logger/ src/logger/
COPY src/exception/ src/exception/
COPY models/vectorizer.pkl models/vectorizer.pkl
COPY app/ .

EXPOSE 8080

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer for Docker cache)
COPY pyproject.toml .
RUN mkdir -p src/llmframe && touch src/llmframe/__init__.py
RUN pip install --no-cache-dir -e ".[all]"

# Copy source (invalidates only when code changes, not deps)
COPY src/ src/

# Runtime directories
RUN mkdir -p data/chroma logs

# Module resolution: run as src.llmframe.api.app from /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.llmframe.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

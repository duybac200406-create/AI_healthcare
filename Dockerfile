
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy package metadata and the README referenced by pyproject.toml.
COPY pyproject.toml README.md ./
COPY docs/ ./docs/
COPY src/ ./src/

# Install the project and its declared runtime dependencies.
RUN python -m pip install --no-cache-dir .

# Copy the Streamlit application and synthetic data.
COPY frontend/ ./frontend/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]

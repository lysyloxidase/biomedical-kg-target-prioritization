FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen --no-editable
RUN groupadd --system kgtp && useradd --system --gid kgtp --home-dir /app kgtp \
    && chmod +x /app/docker/entrypoint.sh

USER kgtp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["/app/.venv/bin/uvicorn", "kgtp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

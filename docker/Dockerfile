FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-editable --all-extras || uv sync --no-editable --all-extras

COPY . .

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["uv", "run", "--no-editable", "uvicorn", "kgtp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

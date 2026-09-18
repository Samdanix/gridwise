# GridWise LLM-assisted energy optimizer -- judge fallback image.
#
# Build:  docker build -t gridwise-api ./backend
# Run:    docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... gridwise-api
FROM python:3.11-slim

# coinor-cbc supplies the CBC solver binary PuLP drives.
RUN apt-get update \
    && apt-get install -y --no-install-recommends coinor-cbc curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=gridwise.settings

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unprivileged.
RUN useradd --create-home --uid 10001 gridwise && chown -R gridwise:gridwise /app
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Bind 0.0.0.0 so the container is reachable from outside.
# Two workers with a generous timeout: the work is I/O-bound on the model
# call, and the judge allows 30s per request.
CMD ["sh", "-c", "gunicorn gridwise.wsgi:application --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 60 --access-logfile - --error-logfile -"]

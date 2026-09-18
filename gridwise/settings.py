"""Django settings for the GridWise API service.

Stateless by design: no database, no sessions, no admin. The service is a
pure request/response optimizer, which keeps cold start fast and the
deployment surface small.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# A key is required by Django but never used for sessions or cookies here.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "gridwise-stateless-service-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "0") == "1"

# Judges call the service from outside; hostname is not a security boundary
# for a public read-only optimizer, and a mismatch would look like an outage.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "optimizer",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    # Public read-only JSON API: permissive CORS so any browser client
    # (including the docs page) can call it. No dependency needed.
    "gridwise.middleware.PermissiveCorsMiddleware",
]

ROOT_URLCONF = "gridwise.urls"
WSGI_APPLICATION = "gridwise.wsgi.application"
TEMPLATES = []
DATABASES = {}
USE_TZ = True
TIME_ZONE = "Asia/Dhaka"
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
APPEND_SLASH = False

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "UNAUTHENTICATED_USER": None,
    "EXCEPTION_HANDLER": "gridwise.errors.exception_handler",
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "plain"}
    },
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
    "loggers": {
        # Never let a library log a request body or an API key.
        "urllib3": {"level": "WARNING"},
        "requests": {"level": "WARNING"},
    },
}

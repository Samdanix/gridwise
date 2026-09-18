"""WSGI entrypoint (gunicorn gridwise.wsgi:application)."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gridwise.settings")

application = get_wsgi_application()

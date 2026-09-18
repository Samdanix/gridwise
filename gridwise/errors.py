"""Uniform, secret-safe error envelopes.

Every failure the judge can trigger -- malformed JSON, wrong method, unknown
path, unhandled exception -- returns a small JSON object. No stack traces, no
HTML debug pages, no configuration values.
"""
import logging

from django.http import JsonResponse
from rest_framework import status
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger(__name__)


def exception_handler(exc, context):
    response = drf_exception_handler(exc, context)
    if response is not None:
        detail = response.data
        if response.status_code == status.HTTP_400_BAD_REQUEST:
            response.data = {"error": "invalid_request", "detail": detail}
        else:
            response.data = {"error": "request_failed", "detail": detail}
        return response

    logger.exception("unhandled exception in %s", context.get("view"))
    return None


def bad_request(request, exception=None):
    return JsonResponse(
        {"error": "invalid_request", "detail": "Malformed request."}, status=400
    )


def not_found(request, exception=None):
    return JsonResponse(
        {
            "error": "not_found",
            "detail": "Unknown endpoint. Use GET /health or POST /optimize-energy.",
        },
        status=404,
    )


def server_error(request):
    return JsonResponse(
        {"error": "internal_error", "detail": "The service failed to complete."},
        status=500,
    )

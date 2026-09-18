"""Minimal CORS support for the public JSON API."""
from django.http import HttpResponse

ALLOW_HEADERS = "Content-Type, Accept"
ALLOW_METHODS = "GET, POST, OPTIONS"


class PermissiveCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse(status=204)
        else:
            response = self.get_response(request)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = ALLOW_HEADERS
        response["Access-Control-Allow-Methods"] = ALLOW_METHODS
        response["Access-Control-Max-Age"] = "86400"
        return response

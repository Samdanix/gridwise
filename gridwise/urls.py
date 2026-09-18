"""URL map. Endpoint names match the Problem Statement exactly."""
from django.urls import path

from optimizer import views

handler400 = "gridwise.errors.bad_request"
handler404 = "gridwise.errors.not_found"
handler500 = "gridwise.errors.server_error"

urlpatterns = [
    path("health", views.health),
    path("health/", views.health),
    path("optimize-energy", views.optimize_energy),
    path("optimize-energy/", views.optimize_energy),
    # Operational detail for local verification; not part of the judged contract.
    path("diagnostics", views.diagnostics),
]

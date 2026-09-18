"""HTTP surface: GET /health and POST /optimize-energy (Section 06)."""
import logging
import time

from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from . import llm
from .pipeline import PipelineError, run
from .serializers import ScenarioSerializer

logger = logging.getLogger(__name__)


@api_view(["GET"])
def health(_request):
    """Readiness probe. Always cheap -- never calls the model."""
    return Response({"status": "ok"}, status=status.HTTP_200_OK)


@api_view(["GET"])
def diagnostics(_request):
    """Non-judged operational detail, useful for local verification."""
    from .solver import pulp

    return Response(
        {
            "status": "ok",
            "llm_configured": llm.is_configured(),
            "model": llm._config()["model"],
            "solver": "CBC via PuLP %s" % pulp.__version__,
        }
    )


@api_view(["POST"])
@parser_classes([JSONParser])
def optimize_energy(request):
    started = time.time()

    serializer = ScenarioSerializer(data=request.data)
    if not serializer.is_valid():
        # Structurally invalid request body -- Section 6.1.
        return Response(
            {"error": "invalid_request", "detail": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    scenario = serializer.to_scenario()

    try:
        payload, meta = run(scenario)
    except PipelineError as exc:
        # Well-formed request that admits no valid schedule.
        logger.warning("scenario %s unsatisfiable: %s", scenario.scenario_id, exc)
        return Response(
            {
                "error": "unsatisfiable_scenario",
                "detail": str(exc),
                "scenario_id": scenario.scenario_id,
            },
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    except Exception:
        # Controlled internal error: no stack trace, no secrets.
        logger.exception("scenario %s failed", scenario.scenario_id)
        return Response(
            {"error": "internal_error", "detail": "The optimizer failed to complete."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    logger.info(
        "scenario %s ok in %dms (llm=%s degraded=%s method=%s)",
        scenario.scenario_id,
        int((time.time() - started) * 1000),
        meta.get("llm_used"),
        meta.get("degraded"),
        meta.get("method"),
    )
    return Response(payload, status=status.HTTP_200_OK)

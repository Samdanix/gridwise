"""Request validation for POST /optimize-energy (Section 07).

Structurally invalid requests are rejected with HTTP 400 before any model
call is made; well-formed but semantically impossible requests get 422.
"""
import math

from rest_framework import serializers

from .schema import HORIZON, Battery, HourInput, Scenario


def _finite(value, field):
    if value is None or not math.isfinite(float(value)):
        raise serializers.ValidationError({field: "must be a finite number"})
    return float(value)


class HourSerializer(serializers.Serializer):
    hour = serializers.IntegerField(min_value=0, max_value=HORIZON - 1)
    demand_kwh = serializers.FloatField(min_value=0)
    solar_kwh = serializers.FloatField(min_value=0)
    tariff_bdt_per_kwh = serializers.FloatField()

    def validate_tariff_bdt_per_kwh(self, value):
        if not math.isfinite(value):
            raise serializers.ValidationError("must be finite")
        return value


class BatterySerializer(serializers.Serializer):
    capacity_kwh = serializers.FloatField(min_value=0)
    initial_energy_kwh = serializers.FloatField(min_value=0)
    minimum_energy_kwh = serializers.FloatField(min_value=0)
    max_charge_kwh_per_hour = serializers.FloatField(min_value=0)
    max_discharge_kwh_per_hour = serializers.FloatField(min_value=0)

    def validate(self, attrs):
        capacity = attrs["capacity_kwh"]
        if attrs["initial_energy_kwh"] > capacity:
            raise serializers.ValidationError(
                "initial_energy_kwh cannot exceed capacity_kwh"
            )
        if attrs["minimum_energy_kwh"] > capacity:
            raise serializers.ValidationError(
                "minimum_energy_kwh cannot exceed capacity_kwh"
            )
        if attrs["initial_energy_kwh"] < attrs["minimum_energy_kwh"]:
            raise serializers.ValidationError(
                "initial_energy_kwh cannot start below minimum_energy_kwh"
            )
        return attrs


class ScenarioSerializer(serializers.Serializer):
    scenario_id = serializers.CharField(max_length=200, allow_blank=False)
    operator_notes = serializers.ListField(
        child=serializers.CharField(allow_blank=False, trim_whitespace=True),
        min_length=1,
        max_length=3,
    )
    hours = serializers.ListField(
        child=HourSerializer(), min_length=HORIZON, max_length=HORIZON
    )
    battery = BatterySerializer()

    def validate_hours(self, value):
        seen = sorted(entry["hour"] for entry in value)
        if seen != list(range(HORIZON)):
            raise serializers.ValidationError(
                "hours must contain each integer 0..%d exactly once" % (HORIZON - 1)
            )
        return value

    def validate_operator_notes(self, value):
        # Enforce that all operator notes are strictly strings, not just castable.
        for note in self.initial_data.get("operator_notes", []):
            if not isinstance(note, str):
                raise serializers.ValidationError("each operator note must be a string")
        return value

    def to_scenario(self) -> Scenario:
        data = self.validated_data
        hours = sorted(data["hours"], key=lambda entry: entry["hour"])
        return Scenario(
            scenario_id=data["scenario_id"],
            operator_notes=list(data["operator_notes"]),
            hours=[
                HourInput(
                    hour=entry["hour"],
                    demand_kwh=float(entry["demand_kwh"]),
                    solar_kwh=float(entry["solar_kwh"]),
                    tariff_bdt_per_kwh=float(entry["tariff_bdt_per_kwh"]),
                )
                for entry in hours
            ],
            battery=Battery(
                capacity_kwh=float(data["battery"]["capacity_kwh"]),
                initial_energy_kwh=float(data["battery"]["initial_energy_kwh"]),
                minimum_energy_kwh=float(data["battery"]["minimum_energy_kwh"]),
                max_charge_kwh_per_hour=float(
                    data["battery"]["max_charge_kwh_per_hour"]
                ),
                max_discharge_kwh_per_hour=float(
                    data["battery"]["max_discharge_kwh_per_hour"]
                ),
            ),
        )

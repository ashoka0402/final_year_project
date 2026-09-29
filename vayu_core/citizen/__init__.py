"""Citizen-sourced observation layer.

Photographs and low-cost sensor readings from the public, turned into structured
observations by Gemini vision and then cross-checked against the satellite
record before they are allowed to influence anything.
"""

from .crosscheck import (
    CONTRADICTED,
    CORROBORATED,
    NO_SATELLITE_DATA,
    UNSUPPORTED,
    Corroboration,
    corroborate,
)

__all__ = [
    "corroborate",
    "Corroboration",
    "CORROBORATED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "NO_SATELLITE_DATA",
]

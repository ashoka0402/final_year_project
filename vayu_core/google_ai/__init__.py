"""Google AI (Gemini) integration.

Two jobs that nothing else in VAYU can do:

  * `vision`  — read a citizen's photograph and return a *structured* pollution
                observation (haze severity, visible source, confidence).
  * `advisory`— write the plain-language health advisory in the reader's own
                language, grounded in numbers we computed, never invented.

Both degrade to an explicit `unavailable` rather than a plausible-looking guess
when no key is configured. That is deliberate: a fabricated pollution reading
would flow into hotspot detection and corrupt the science, which is strictly
worse than having no reading at all.
"""

from .client import GeminiUnavailable, available, generate, generate_json
from .vision import PhotoObservation, analyse_photo

__all__ = [
    "available",
    "generate",
    "generate_json",
    "GeminiUnavailable",
    "analyse_photo",
    "PhotoObservation",
]

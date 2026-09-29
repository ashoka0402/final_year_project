"""Herald — turns a forecast into advice a person can act on (PRD Epic D).

The Command Center speaks to a commissioner; the Herald speaks to a parent
deciding whether their child walks to school. Same forecast, different language —
literally: advisories ship in English, Hindi, and Punjabi so the citizen page
meets the ≥3-language coverage the PRD asks for, and so the advice reaches the
people actually breathing the air.

Two products here:
  * `clean_hours` — the 48h AQI trajectory and the best window to be outside.
  * `advisory` — a ≤80-word, audience- and language-specific message.

Advisories are deterministic templates, not LLM output. TRD 8 makes the LLM
optional precisely so the demo works offline and a public-health message is never
a hallucination — the template is the fallback and, here, the default. Every
message is keyed to a real severity bucket and carries its provenance line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np

from vayu_core.aqi import category_for

LANGUAGES = ("en", "hi", "pa")
LANGUAGE_LABEL = {"en": "English", "hi": "हिंदी", "pa": "ਪੰਜਾਬੀ"}

AUDIENCES = ("general", "children", "sensitive", "outdoor")
AUDIENCE_LABEL = {
    "en": {"general": "Everyone", "children": "Children & schools",
           "sensitive": "Elderly & respiratory", "outdoor": "Outdoor workers"},
    "hi": {"general": "सभी के लिए", "children": "बच्चे और स्कूल",
           "sensitive": "बुज़ुर्ग और साँस के रोगी", "outdoor": "बाहर काम करने वाले"},
    "pa": {"general": "ਸਾਰਿਆਂ ਲਈ", "children": "ਬੱਚੇ ਅਤੇ ਸਕੂਲ",
           "sensitive": "ਬਜ਼ੁਰਗ ਅਤੇ ਸਾਹ ਦੇ ਰੋਗੀ", "outdoor": "ਬਾਹਰ ਕੰਮ ਕਰਨ ਵਾਲੇ"},
}

# AQI severity bucket. Mirrors the CPCB categories but grouped by the ACTION a
# citizen takes, which is what an advisory is for.
def _bucket(aqi: int) -> str:
    if aqi <= 100:
        return "ok"          # Good / Satisfactory
    if aqi <= 200:
        return "moderate"
    if aqi <= 300:
        return "poor"
    if aqi <= 400:
        return "very_poor"
    return "severe"


# Templates: [language][bucket][audience]. Kept short (<=80 words), concrete, and
# translated rather than transliterated. {best} is filled with the clean window.
_T: dict = {
    "en": {
        "ok": {
            "general": "Air quality is acceptable today. Normal outdoor activity is fine for everyone.",
            "children": "Children can play and commute outdoors as usual today.",
            "sensitive": "Air is clean enough for normal activity. Keep any prescribed inhaler handy as routine.",
            "outdoor": "Safe for a full outdoor shift. Stay hydrated.",
        },
        "moderate": {
            "general": "Air is moderate. Most people are fine outdoors; cut back on prolonged heavy exertion. Best window: {best}.",
            "children": "School activity is fine, but move long sports sessions to {best} when air is cleanest.",
            "sensitive": "You may notice mild discomfort. Keep outdoor trips short and prefer {best}. Carry your inhaler.",
            "outdoor": "Take breaks indoors where possible. Schedule the heaviest work for {best}.",
        },
        "poor": {
            "general": "Air is poor. Reduce time outdoors and avoid heavy exertion. The cleanest window is {best}.",
            "children": "Keep PE and outdoor play indoors today. If children must go out, use {best} and a mask.",
            "sensitive": "Stay indoors as much as you can. Keep windows shut and use a mask outside. Follow your medication plan.",
            "outdoor": "Wear an N95, take frequent indoor breaks, and shift strenuous tasks to {best}.",
        },
        "very_poor": {
            "general": "Air is very poor. Stay indoors, keep windows closed, and wear an N95 if you must go out. Avoid exercise outdoors.",
            "children": "Outdoor activities should be cancelled. Keep children indoors; consider a purifier in classrooms.",
            "sensitive": "Remain indoors. Run a purifier if available, keep medication close, and seek care if breathing worsens.",
            "outdoor": "Only essential outdoor work, always in an N95, with regular indoor breaks. Employers should limit exposure.",
        },
        "severe": {
            "general": "Air is severe — a health emergency. Stay indoors, seal windows, run a purifier, and do not exercise outside.",
            "children": "Keep children indoors entirely. Schools should consider closure or a full indoor day.",
            "sensitive": "Do not go outside. Use a purifier, follow your action plan, and seek medical help for any chest tightness or breathlessness.",
            "outdoor": "Outdoor work should be suspended except emergencies. N95 is mandatory; employers must relocate work indoors.",
        },
    },
    "hi": {
        "ok": {
            "general": "आज हवा की गुणवत्ता ठीक है। सभी लोग सामान्य रूप से बाहर की गतिविधियाँ कर सकते हैं।",
            "children": "बच्चे आज हमेशा की तरह बाहर खेल और आ-जा सकते हैं।",
            "sensitive": "हवा सामान्य गतिविधि के लिए पर्याप्त साफ़ है। अपना इनहेलर हमेशा की तरह पास रखें।",
            "outdoor": "पूरी पाली बाहर काम करना सुरक्षित है। पानी पीते रहें।",
        },
        "moderate": {
            "general": "हवा मध्यम है। अधिकतर लोग बाहर ठीक रहेंगे; लंबे भारी परिश्रम से बचें। सबसे अच्छा समय: {best}।",
            "children": "स्कूल की गतिविधि ठीक है, पर लंबे खेल-सत्र {best} पर रखें जब हवा सबसे साफ़ हो।",
            "sensitive": "हल्की तकलीफ़ हो सकती है। बाहर कम समय रहें और {best} को प्राथमिकता दें। इनहेलर साथ रखें।",
            "outdoor": "जहाँ संभव हो अंदर विश्राम लें। सबसे भारी काम {best} के लिए रखें।",
        },
        "poor": {
            "general": "हवा ख़राब है। बाहर कम समय बिताएँ और भारी परिश्रम से बचें। सबसे साफ़ समय {best} है।",
            "children": "आज पीटी और बाहरी खेल अंदर ही रखें। बच्चों को बाहर जाना ही हो तो {best} पर और मास्क के साथ।",
            "sensitive": "जितना हो सके अंदर रहें। खिड़कियाँ बंद रखें और बाहर मास्क पहनें। दवा योजना का पालन करें।",
            "outdoor": "एन95 पहनें, बार-बार अंदर विश्राम लें, और कठिन काम {best} पर करें।",
        },
        "very_poor": {
            "general": "हवा बहुत ख़राब है। अंदर रहें, खिड़कियाँ बंद रखें, और बाहर जाना ज़रूरी हो तो एन95 पहनें। बाहर व्यायाम न करें।",
            "children": "बाहरी गतिविधियाँ रद्द करें। बच्चों को अंदर रखें; कक्षाओं में प्यूरिफ़ायर पर विचार करें।",
            "sensitive": "अंदर ही रहें। प्यूरिफ़ायर चलाएँ, दवा पास रखें, और साँस बिगड़ने पर चिकित्सा लें।",
            "outdoor": "केवल ज़रूरी बाहरी काम, हमेशा एन95 में, नियमित अंदरूनी विश्राम के साथ। नियोक्ता जोखिम सीमित करें।",
        },
        "severe": {
            "general": "हवा गंभीर है — स्वास्थ्य आपातकाल। अंदर रहें, खिड़कियाँ सील करें, प्यूरिफ़ायर चलाएँ, बाहर व्यायाम न करें।",
            "children": "बच्चों को पूरी तरह अंदर रखें। स्कूल बंद करने या पूरे दिन अंदर रखने पर विचार करें।",
            "sensitive": "बाहर न जाएँ। प्यूरिफ़ायर चलाएँ, अपनी योजना का पालन करें, और सीने में जकड़न या साँस फूलने पर तुरंत चिकित्सा लें।",
            "outdoor": "आपात को छोड़कर बाहरी काम रोक दें। एन95 अनिवार्य; नियोक्ता काम अंदर कराएँ।",
        },
    },
    "pa": {
        "ok": {
            "general": "ਅੱਜ ਹਵਾ ਦੀ ਗੁਣਵੱਤਾ ਠੀਕ ਹੈ। ਸਾਰੇ ਆਮ ਵਾਂਗ ਬਾਹਰ ਦੀਆਂ ਗਤੀਵਿਧੀਆਂ ਕਰ ਸਕਦੇ ਹਨ।",
            "children": "ਬੱਚੇ ਅੱਜ ਆਮ ਵਾਂਗ ਬਾਹਰ ਖੇਡ ਤੇ ਆ-ਜਾ ਸਕਦੇ ਹਨ।",
            "sensitive": "ਹਵਾ ਆਮ ਗਤੀਵਿਧੀ ਲਈ ਕਾਫ਼ੀ ਸਾਫ਼ ਹੈ। ਆਪਣਾ ਇਨਹੇਲਰ ਆਮ ਵਾਂਗ ਕੋਲ ਰੱਖੋ।",
            "outdoor": "ਪੂਰੀ ਸ਼ਿਫ਼ਟ ਬਾਹਰ ਕੰਮ ਕਰਨਾ ਸੁਰੱਖਿਅਤ ਹੈ। ਪਾਣੀ ਪੀਂਦੇ ਰਹੋ।",
        },
        "moderate": {
            "general": "ਹਵਾ ਦਰਮਿਆਨੀ ਹੈ। ਬਹੁਤੇ ਲੋਕ ਬਾਹਰ ਠੀਕ ਰਹਿਣਗੇ; ਲੰਮੀ ਸਖ਼ਤ ਮਿਹਨਤ ਤੋਂ ਬਚੋ। ਵਧੀਆ ਸਮਾਂ: {best}।",
            "children": "ਸਕੂਲ ਦੀ ਗਤੀਵਿਧੀ ਠੀਕ ਹੈ, ਪਰ ਲੰਮੇ ਖੇਡ-ਸੈਸ਼ਨ {best} ਵੇਲੇ ਰੱਖੋ ਜਦੋਂ ਹਵਾ ਸਭ ਤੋਂ ਸਾਫ਼ ਹੋਵੇ।",
            "sensitive": "ਹਲਕੀ ਤਕਲੀਫ਼ ਹੋ ਸਕਦੀ ਹੈ। ਬਾਹਰ ਘੱਟ ਸਮਾਂ ਰਹੋ ਤੇ {best} ਨੂੰ ਤਰਜੀਹ ਦਿਓ। ਇਨਹੇਲਰ ਨਾਲ ਰੱਖੋ।",
            "outdoor": "ਜਿੱਥੇ ਹੋ ਸਕੇ ਅੰਦਰ ਆਰਾਮ ਕਰੋ। ਸਭ ਤੋਂ ਸਖ਼ਤ ਕੰਮ {best} ਲਈ ਰੱਖੋ।",
        },
        "poor": {
            "general": "ਹਵਾ ਖ਼ਰਾਬ ਹੈ। ਬਾਹਰ ਘੱਟ ਸਮਾਂ ਰਹੋ ਤੇ ਸਖ਼ਤ ਮਿਹਨਤ ਤੋਂ ਬਚੋ। ਸਭ ਤੋਂ ਸਾਫ਼ ਸਮਾਂ {best} ਹੈ।",
            "children": "ਅੱਜ ਪੀ.ਟੀ. ਤੇ ਬਾਹਰੀ ਖੇਡ ਅੰਦਰ ਹੀ ਰੱਖੋ। ਬੱਚਿਆਂ ਨੂੰ ਬਾਹਰ ਜਾਣਾ ਪਵੇ ਤਾਂ {best} ਵੇਲੇ ਤੇ ਮਾਸਕ ਨਾਲ।",
            "sensitive": "ਜਿੰਨਾ ਹੋ ਸਕੇ ਅੰਦਰ ਰਹੋ। ਖਿੜਕੀਆਂ ਬੰਦ ਰੱਖੋ ਤੇ ਬਾਹਰ ਮਾਸਕ ਪਾਓ। ਦਵਾਈ ਦੀ ਯੋਜਨਾ ਦੀ ਪਾਲਣਾ ਕਰੋ।",
            "outdoor": "ਐਨ95 ਪਾਓ, ਵਾਰ-ਵਾਰ ਅੰਦਰ ਆਰਾਮ ਕਰੋ, ਤੇ ਔਖਾ ਕੰਮ {best} ਵੇਲੇ ਕਰੋ।",
        },
        "very_poor": {
            "general": "ਹਵਾ ਬਹੁਤ ਖ਼ਰਾਬ ਹੈ। ਅੰਦਰ ਰਹੋ, ਖਿੜਕੀਆਂ ਬੰਦ ਰੱਖੋ, ਤੇ ਬਾਹਰ ਜਾਣਾ ਜ਼ਰੂਰੀ ਹੋਵੇ ਤਾਂ ਐਨ95 ਪਾਓ। ਬਾਹਰ ਕਸਰਤ ਨਾ ਕਰੋ।",
            "children": "ਬਾਹਰੀ ਗਤੀਵਿਧੀਆਂ ਰੱਦ ਕਰੋ। ਬੱਚਿਆਂ ਨੂੰ ਅੰਦਰ ਰੱਖੋ; ਕਲਾਸਾਂ ਵਿੱਚ ਪਿਊਰੀਫ਼ਾਇਰ ਬਾਰੇ ਸੋਚੋ।",
            "sensitive": "ਅੰਦਰ ਹੀ ਰਹੋ। ਪਿਊਰੀਫ਼ਾਇਰ ਚਲਾਓ, ਦਵਾਈ ਕੋਲ ਰੱਖੋ, ਤੇ ਸਾਹ ਵਿਗੜਨ 'ਤੇ ਇਲਾਜ ਲਓ।",
            "outdoor": "ਸਿਰਫ਼ ਜ਼ਰੂਰੀ ਬਾਹਰੀ ਕੰਮ, ਹਮੇਸ਼ਾ ਐਨ95 ਵਿੱਚ, ਨਿਯਮਤ ਅੰਦਰੂਨੀ ਆਰਾਮ ਨਾਲ। ਮਾਲਕ ਖ਼ਤਰਾ ਘਟਾਉਣ।",
        },
        "severe": {
            "general": "ਹਵਾ ਗੰਭੀਰ ਹੈ — ਸਿਹਤ ਐਮਰਜੈਂਸੀ। ਅੰਦਰ ਰਹੋ, ਖਿੜਕੀਆਂ ਸੀਲ ਕਰੋ, ਪਿਊਰੀਫ਼ਾਇਰ ਚਲਾਓ, ਬਾਹਰ ਕਸਰਤ ਨਾ ਕਰੋ।",
            "children": "ਬੱਚਿਆਂ ਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਅੰਦਰ ਰੱਖੋ। ਸਕੂਲ ਬੰਦ ਕਰਨ ਜਾਂ ਪੂਰਾ ਦਿਨ ਅੰਦਰ ਰੱਖਣ ਬਾਰੇ ਸੋਚਣ।",
            "sensitive": "ਬਾਹਰ ਨਾ ਜਾਓ। ਪਿਊਰੀਫ਼ਾਇਰ ਚਲਾਓ, ਆਪਣੀ ਯੋਜਨਾ ਦੀ ਪਾਲਣਾ ਕਰੋ, ਤੇ ਛਾਤੀ 'ਚ ਜਕੜਨ ਜਾਂ ਸਾਹ ਚੜ੍ਹਨ 'ਤੇ ਤੁਰੰਤ ਇਲਾਜ ਲਓ।",
            "outdoor": "ਐਮਰਜੈਂਸੀ ਤੋਂ ਬਿਨਾਂ ਬਾਹਰੀ ਕੰਮ ਰੋਕੋ। ਐਨ95 ਲਾਜ਼ਮੀ; ਮਾਲਕ ਕੰਮ ਅੰਦਰ ਕਰਵਾਉਣ।",
        },
    },
}

# The provenance line, per language. Every advisory carries where it came from.
_SOURCE = {
    "en": "VAYU forecast · updated {t}",
    "hi": "वायु पूर्वानुमान · अद्यतन {t}",
    "pa": "ਵਾਯੂ ਪੂਰਵ-ਅਨੁਮਾਨ · ਅੱਪਡੇਟ {t}",
}

_NO_CLEAN = {"en": "no clearly cleaner window in 48h",
             "hi": "48 घंटे में कोई साफ़ समय नहीं",
             "pa": "48 ਘੰਟਿਆਂ 'ਚ ਕੋਈ ਸਾਫ਼ ਸਮਾਂ ਨਹੀਂ"}


@dataclass
class HourBlock:
    ts: str
    aqi: int
    category: str
    color: str
    clean: bool


@dataclass
class CleanHours:
    blocks: list[HourBlock]
    best_window: str | None       # e.g. "06:00–09:00, 4 Nov" (local)
    best_window_start: str | None
    best_aqi: int | None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Advisory:
    audience: str
    audience_label: str
    text: str
    source: str


@dataclass
class CitizenBrief:
    ward_id: str
    ward_name: str
    language: str
    now_aqi: int | None
    now_category: str | None
    now_color: str | None
    clean_hours: CleanHours
    advisories: list[Advisory] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ward_id": self.ward_id,
            "ward_name": self.ward_name,
            "language": self.language,
            "now_aqi": self.now_aqi,
            "now_category": self.now_category,
            "now_color": self.now_color,
            "clean_hours": self.clean_hours.to_dict(),
            "advisories": [asdict(a) for a in self.advisories],
        }


def _interpolate_hourly(now_aqi: float, anchors: dict[int, float], at: datetime, hours: int = 48) -> list[tuple[datetime, float]]:
    """Hourly AQI from the discrete forecast anchors (t+0, +24, +48, +72).

    The forecaster stores three horizons, not a continuous curve, so the clean-
    hours strip is a linear interpolation between them anchored on the current
    reading. This is presented as guidance ("best window"), not a per-hour
    promise, and the Methodology page says so.
    """
    known_h = sorted({0, *anchors.keys()})
    known_v = [now_aqi if h == 0 else anchors[h] for h in known_h]
    xs = np.arange(0, hours + 1)
    ys = np.interp(xs, known_h, known_v)
    return [(at + timedelta(hours=int(h)), float(v)) for h, v in zip(xs, ys)]


def clean_hours(now_aqi: float | None, anchors: dict[int, float], at: datetime, tz: str) -> CleanHours:
    """48h trajectory + the cleanest 3-hour window to be outside."""
    from zoneinfo import ZoneInfo

    if now_aqi is None or not anchors:
        return CleanHours(blocks=[], best_window=None, best_window_start=None, best_aqi=None)

    zone = ZoneInfo(tz)
    series = _interpolate_hourly(float(now_aqi), anchors, at)

    # A block is "clean" if it's in the best third of the next 48h AND genuinely
    # breathable-ish (<=200). Marking a 260 block green just because it's the
    # day's low would be dishonest.
    vals = np.array([v for _, v in series])
    threshold = min(float(np.percentile(vals, 25)), 200.0)

    blocks: list[HourBlock] = []
    for ts, v in series:
        aqi = int(round(v))
        label, color = category_for(aqi)
        blocks.append(HourBlock(
            ts=ts.astimezone(zone).isoformat(), aqi=aqi, category=label,
            color=color, clean=v <= threshold,
        ))

    # Best contiguous 3h window by mean AQI.
    best_start, best_mean = None, float("inf")
    for i in range(len(series) - 2):
        m = float(np.mean([series[i][1], series[i + 1][1], series[i + 2][1]]))
        if m < best_mean:
            best_mean, best_start = m, series[i][0]

    best_window = None
    best_window_start = None
    if best_start is not None and best_mean <= 200.0:
        s = best_start.astimezone(zone)
        e = (best_start + timedelta(hours=3)).astimezone(zone)
        best_window = f"{s:%H:%M}–{e:%H:%M}, {s:%-d %b}"
        best_window_start = s.isoformat()

    return CleanHours(
        blocks=blocks, best_window=best_window, best_window_start=best_window_start,
        best_aqi=int(round(best_mean)) if best_start is not None else None,
    )


def advisory(aqi: int | None, audience: str, language: str, best_window: str | None,
             updated_local: str) -> Advisory:
    """One audience/language advisory. Always returns something usable."""
    lang = language if language in LANGUAGES else "en"
    aud = audience if audience in AUDIENCES else "general"
    bucket = _bucket(aqi) if aqi is not None else "moderate"

    best = best_window or _NO_CLEAN[lang]
    text = _T[lang][bucket][aud].replace("{best}", best)
    source = _SOURCE[lang].replace("{t}", updated_local)
    return Advisory(audience=aud, audience_label=AUDIENCE_LABEL[lang][aud],
                    text=text, source=source)


def brief(
    ward_id: str, ward_name: str, now_aqi: int | None, anchors: dict[int, float],
    at: datetime, tz: str, language: str = "en",
) -> CitizenBrief:
    """The whole citizen payload for one ward, in one language."""
    from zoneinfo import ZoneInfo

    lang = language if language in LANGUAGES else "en"
    ch = clean_hours(now_aqi, anchors, at, tz)
    updated = at.astimezone(ZoneInfo(tz)).strftime("%H:%M, %-d %b")

    cat, color = (category_for(now_aqi) if now_aqi is not None else (None, None))
    advisories = [
        advisory(now_aqi, aud, lang, ch.best_window, updated) for aud in AUDIENCES
    ]
    return CitizenBrief(
        ward_id=ward_id, ward_name=ward_name, language=lang,
        now_aqi=now_aqi, now_category=cat, now_color=color,
        clean_hours=ch, advisories=advisories,
    )

#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Deepdub endpoints, models, voices and languages."""

DEFAULT_TTS_URL = "wss://wss.deepdub.ai/ws"
"""US streaming text-to-speech endpoint. The EU endpoint is ``wss://wss.eu.deepdub.ai/ws``."""

DEFAULT_STT_URL = "wss://wsgw.deepdub.ai/transcribe-websocket"
"""Real-time speech-to-text endpoint (US only)."""

DEFAULT_TTS_MODEL = "dd-etts-4.0"
DEFAULT_STT_MODEL = "stt-egf"

DEFAULT_VOICE_PROMPT_ID = "b2abb241-ac92-48bf-a890-fec03f43e209"
"""English preset "Call Center Agent (Neutral)". See https://docs.deepdub.ai/voice-presets."""

DEFAULT_LOCALE = "en-US"

TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 36000, 44100, 48000)
"""Output sample rates Deepdub can stream, in Hz."""

TTS_LOCALES = frozenset(
    {
        "ar-LB",
        "ar-QA",
        "ar-SA",
        "ar-SY",
        "cs-CZ",
        "da-DK",
        "de-DE",
        "el-GR",
        "en-AU",
        "en-GB",
        "en-US",
        "es-419",
        "es-ES",
        "es-MX",
        "et-EE",
        "fi-FI",
        "fr-FR",
        "he-IL",
        "hi-IN",
        "hu-HU",
        "id-ID",
        "it-IT",
        "ja-JP",
        "ko-KR",
        "mk-MK",
        "nb-NO",
        "nl-NL",
        "pl-PL",
        "pt-BR",
        "ro-RO",
        "ru-RU",
        "sv-SE",
        "ta-IN",
        "th-TH",
        "tr-TR",
    }
)
"""Locales Deepdub text-to-speech can speak."""

DEFAULT_LOCALE_FOR_LANGUAGE = {
    "ar": "ar-SA",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "en": "en-US",
    "es": "es-ES",
    "et": "et-EE",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "he": "he-IL",
    "hi": "hi-IN",
    "hu": "hu-HU",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "mk": "mk-MK",
    "nb": "nb-NO",
    "no": "nb-NO",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-BR",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sv": "sv-SE",
    "ta": "ta-IN",
    "th": "th-TH",
    "tr": "tr-TR",
}
"""Locale used when only a base language such as ``"en"`` is given."""

STT_LANGUAGES = frozenset({"en", "he"})
"""Languages Deepdub speech-to-text recognizes."""


def locale_for(language: str) -> str | None:
    """Return the Deepdub text-to-speech locale for a language or locale code.

    Args:
        language: A code such as ``"en-US"``, ``"en_us"`` or ``"he"``.

    Returns:
        The Deepdub locale, or None if Deepdub doesn't speak the language.
    """
    parts = language.replace("_", "-").split("-")
    code = parts[0].lower()
    if len(parts) > 1:
        region = parts[1] if parts[1].isdigit() else parts[1].upper()
        locale = f"{code}-{region}"
        if locale in TTS_LOCALES:
            return locale
    return DEFAULT_LOCALE_FOR_LANGUAGE.get(code)


def stt_language_for(language: str) -> str | None:
    """Return the Deepdub speech-to-text language for a language or locale code.

    Args:
        language: A code such as ``"he-IL"`` or ``"en"``.

    Returns:
        ``"en"`` or ``"he"``, or None if Deepdub doesn't recognize the language.
    """
    code = language.replace("_", "-").split("-")[0].lower()
    return code if code in STT_LANGUAGES else None

#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Deepdub text-to-speech and speech-to-text services for Pipecat."""

from pipecat_deepdub.stt import DeepdubSTTService, DeepdubSTTSettings
from pipecat_deepdub.tts import DeepdubTTSService, DeepdubTTSSettings

__all__ = [
    "DeepdubSTTService",
    "DeepdubSTTSettings",
    "DeepdubTTSService",
    "DeepdubTTSSettings",
]

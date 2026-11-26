#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""DeepdubSTTService against a fake of Deepdub's real-time speech-to-text endpoint."""

import json

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from websockets.asyncio.server import serve

from pipecat_deepdub import DeepdubSTTService

# 20 ms of 16 kHz 16-bit mono audio.
AUDIO = b"\x10\x00" * 320


def _token(text: str, *, final: bool, lang: str = "en", start: int = 0, end: int = 0) -> dict:
    return {
        "text": text,
        "is_final": final,
        "language": lang,
        "start_ms": start,
        "end_ms": end,
        "confidence": 0.9,
    }


def _server(captured: dict, script):
    """Fake endpoint: reads the config, then hands each message to ``script``."""

    async def handler(ws):
        captured["connections"] = captured.get("connections", 0) + 1
        captured["auth"] = ws.request.headers.get("Authorization")
        captured.setdefault("configs", []).append(json.loads(await ws.recv()))
        captured.setdefault("audio", 0)
        captured.setdefault("control", [])
        async for message in ws:
            if isinstance(message, bytes):
                captured["audio"] += len(message)
            elif message:
                captured["control"].append(json.loads(message)["type"])
            await script(ws, message, captured)

    return handler


def _service(server, **kwargs) -> DeepdubSTTService:
    port = next(iter(server.sockets)).getsockname()[1]
    return DeepdubSTTService(
        api_key="dd-test", url=f"ws://127.0.0.1:{port}/transcribe-websocket", **kwargs
    )


def _audio(count: int) -> list[InputAudioRawFrame]:
    return [InputAudioRawFrame(audio=AUDIO, sample_rate=16000, num_channels=1)] * count


async def _speak_on_first_audio(ws, message, captured):
    if isinstance(message, bytes) and not captured.get("spoke"):
        captured["spoke"] = True
        await ws.send(json.dumps({"tokens": [_token("Hel", final=False)]}))
        await ws.send(json.dumps({"tokens": [_token("Hello,", final=True, end=500)]}))
        await ws.send(
            json.dumps(
                {
                    "tokens": [
                        _token(" world.", final=True, start=600, end=1000),
                        _token("<end>", final=True),
                    ]
                }
            )
        )


async def test_streams_interim_and_final_transcripts():
    captured: dict = {}
    async with serve(_server(captured, _speak_on_first_audio), "127.0.0.1", 0) as server:
        service = _service(
            server,
            settings=DeepdubSTTService.Settings(
                language=Language.EN_US, context_terms=["Deepdub"], max_endpoint_delay_ms=700
            ),
        )
        down, up = await run_test(service, frames_to_send=[*_audio(5), SleepFrame(sleep=0.3)])

    assert captured["auth"] == "Bearer dd-test"
    assert captured["configs"][0] == {
        "model": "stt-egf",
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "enable_endpoint_detection": True,
        "language_hints": ["en"],
        "max_endpoint_delay_ms": 700,
        "context": {"terms": ["Deepdub"]},
    }
    assert captured["audio"] == 5 * len(AUDIO)
    interims = [f.text for f in down if isinstance(f, InterimTranscriptionFrame)]
    assert interims == ["Hel", "Hello,"]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in finals] == ["Hello, world."]
    assert finals[0].language == Language.EN
    assert finals[0].finalized
    assert not any(isinstance(f, ErrorFrame) for f in up)


async def test_vad_stop_finalizes_the_transcript():
    async def script(ws, message, captured):
        if message == json.dumps({"type": "finalize"}):
            await ws.send(
                json.dumps(
                    {
                        "tokens": [
                            _token("שלום", final=True, lang="he", end=400),
                            _token("<fin>", final=True),
                        ]
                    }
                )
            )

    captured: dict = {}
    async with serve(_server(captured, script), "127.0.0.1", 0) as server:
        down, _ = await run_test(
            _service(server),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                *_audio(3),
                VADUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.3),
            ],
        )

    assert "language_hints" not in captured["configs"][0]
    assert captured["control"].count("finalize") == 1
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in finals] == ["שלום"]
    assert finals[0].language == Language.HE
    assert finals[0].finalized


async def test_silence_tokens_do_not_reopen_the_turn():
    # The message sequence Deepdub sent live: empty non-final tokens while it hears
    # silence, including one alongside the finalize confirmation.
    async def script(ws, message, captured):
        if isinstance(message, bytes) and not captured.get("spoke"):
            captured["spoke"] = True
            await ws.send(json.dumps({"tokens": [_token("", final=False)]}))
            await ws.send(json.dumps({"tokens": [_token("Hi,", final=False)]}))
            await ws.send(
                json.dumps({"tokens": [_token("Hi,", final=True), _token(" there", final=False)]})
            )
        elif message == json.dumps({"type": "finalize"}):
            await ws.send(
                json.dumps(
                    {
                        "tokens": [
                            _token(" there", final=True),
                            _token("<fin>", final=True),
                            _token("", final=False),
                        ]
                    }
                )
            )
            await ws.send(json.dumps({"tokens": [_token("", final=False)]}))

    captured: dict = {}
    async with serve(_server(captured, script), "127.0.0.1", 0) as server:
        down, _ = await run_test(
            _service(server),
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                *_audio(3),
                VADUserStoppedSpeakingFrame(),
                *_audio(3),
                SleepFrame(sleep=0.3),
            ],
        )

    transcripts = [
        f for f in down if isinstance(f, (InterimTranscriptionFrame, TranscriptionFrame))
    ]
    assert [(type(f).__name__, f.text) for f in transcripts] == [
        ("InterimTranscriptionFrame", "Hi,"),
        ("InterimTranscriptionFrame", "Hi, there"),
        ("TranscriptionFrame", "Hi, there"),
    ]
    assert transcripts[-1].finalized


async def test_deepdub_turn_detection_proposes_turns():
    captured: dict = {}
    async with serve(_server(captured, _speak_on_first_audio), "127.0.0.1", 0) as server:
        service = _service(server, vad_force_turn_endpoint=False)
        down, up = await run_test(service, frames_to_send=[*_audio(3), SleepFrame(sleep=0.3)])

    types = [type(f) for f in down + up]
    assert ProposedUserStartedSpeakingFrame in types
    assert ProposedUserStoppedSpeakingFrame in types
    assert "finalize" not in captured["control"]
    assert isinstance(
        service.service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies
    )


async def test_error_message_is_reported():
    async def script(ws, message, captured):
        if isinstance(message, bytes) and not captured.get("failed"):
            captured["failed"] = True
            await ws.send(
                json.dumps(
                    {"error_code": 401, "error_type": "unauthorized", "error_message": "bad key"}
                )
            )

    captured: dict = {}
    async with serve(_server(captured, script), "127.0.0.1", 0) as server:
        _, up = await run_test(_service(server), frames_to_send=[*_audio(2), SleepFrame(sleep=0.2)])

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert any("401" in f.error and "bad key" in f.error for f in errors)


async def test_settings_update_reconnects_with_the_new_config():
    async def script(ws, message, captured):
        pass

    captured: dict = {}
    async with serve(_server(captured, script), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                STTUpdateSettingsFrame(delta=DeepdubSTTService.Settings(language=Language.HE)),
                SleepFrame(sleep=0.3),
            ],
        )

    assert captured["connections"] == 2
    assert captured["configs"][-1]["language_hints"] == ["he"]


def test_options_validation():
    with pytest.raises(ValueError, match="English and Hebrew"):
        DeepdubSTTService(api_key="dd-test", settings=DeepdubSTTService.Settings(language="fr"))
    with pytest.raises(ValueError, match="max_endpoint_delay_ms"):
        DeepdubSTTService(
            api_key="dd-test", settings=DeepdubSTTService.Settings(max_endpoint_delay_ms=100)
        )
    service = DeepdubSTTService(api_key="dd-test")
    assert service.language_to_service_language(Language.HE_IL) == "he"
    assert service.language_to_service_language(Language.FR) is None

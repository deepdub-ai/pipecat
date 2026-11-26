#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""DeepdubTTSService against a fake of Deepdub's streaming endpoint."""

import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from websockets.asyncio.server import serve

from pipecat_deepdub import DeepdubTTSService

# 50 ms of 24 kHz 16-bit mono audio per chunk.
AUDIO_CHUNK = b"\x01\x00" * 1200


def _chunk(generation_id: str, index: int, **flags) -> bytes:
    """An audio chunk, sent as a binary frame of JSON the way Deepdub sends it."""
    message = {"generationId": generation_id, "index": index, "isFinished": False, **flags}
    if not flags.get("isCancelled"):
        message["data"] = base64.b64encode(AUDIO_CHUNK).decode()
    return json.dumps(message).encode()


def _server(captured: dict, *, hold_first_turn=False, mute=False, error: dict | None = None):
    """Fake Deepdub streaming endpoint.

    Each end-stream is answered, after a short delay, with two chunks, the second
    carrying isFinal. As on the real service, text sent before a turn's isFinal joins
    that turn, which is recorded as a violation. ``hold_first_turn`` speaks one chunk
    of the first turn as soon as text arrives and never finishes it; ``mute`` never
    answers end-stream; ``error`` answers the first end-stream with that message.
    """

    errors = [error] if error else []

    async def handler(ws):
        captured["connections"] = captured.get("connections", 0) + 1
        captured["api_key"] = ws.request.headers.get("x-api-key")
        await ws.send(json.dumps({"action": "status", "message": "connected"}))
        speaking = asyncio.Event()
        turn = 0
        tasks = []

        async def speak(number: int):
            speaking.set()
            await asyncio.sleep(0.05)
            await ws.send(_chunk(f"g{number}", 0))
            await ws.send(_chunk(f"g{number}", 1, isFinished=True, isFinal=True))
            speaking.clear()

        async for raw in ws:
            msg = json.loads(raw)
            captured["messages"].append(msg)
            action = msg["action"]
            if action == "stream-text":
                if speaking.is_set():
                    captured["violations"] = captured.get("violations", 0) + 1
                if hold_first_turn and turn == 0:
                    turn = -1
                    speaking.set()
                    await ws.send(_chunk("held", 0))
            elif action == "end-stream":
                if turn == -1 or mute:
                    continue
                if errors:
                    await ws.send(json.dumps(errors.pop()))
                    continue
                turn += 1
                tasks.append(asyncio.create_task(speak(turn)))
            elif action == "cancel":
                speaking.clear()
                turn = max(turn, 0)
                await ws.send(_chunk("held", 1, isFinished=True, isCancelled=True))
                await ws.send(json.dumps({"action": "status", "message": "cancelled"}))
            elif action == "ping":
                await ws.send(json.dumps({"action": "pong"}))

    return handler


def _service(server, **kwargs) -> DeepdubTTSService:
    port = next(iter(server.sockets)).getsockname()[1]
    return DeepdubTTSService(
        api_key="dd-test", url=f"ws://127.0.0.1:{port}/ws", sample_rate=24000, **kwargs
    )


def _sent(captured: dict, action: str) -> list[dict]:
    return [m for m in captured["messages"] if m["action"] == action]


def _turn_texts(captured: dict) -> list[str]:
    """The text of each turn, as Deepdub received it."""
    turns, text = [], ""
    for msg in captured["messages"]:
        if msg["action"] == "stream-text":
            text += msg["data"]["text"]
        elif msg["action"] == "end-stream":
            turns.append(text)
            text = ""
    return turns


async def test_llm_reply_streams_tokens_and_ends_the_turn():
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        service = _service(
            server,
            enable_logging=False,
            settings=DeepdubTTSService.Settings(voice="voice-1", language=Language.HE_IL),
        )
        down, up = await run_test(
            service,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame(text="Hello "),
                LLMTextFrame(text="there."),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
        )

    assert captured["api_key"] == "dd-test"
    assert captured["connections"] == 1
    assert captured["messages"][0] == {
        "action": "stream-config",
        "data": {
            "model": "dd-etts-4.0",
            "voicePromptId": "voice-1",
            "locale": "he-IL",
            "format": "s16le",
            "sampleRate": 24000,
            "enableLogging": False,
        },
    }
    assert [m["data"]["text"] for m in _sent(captured, "stream-text")] == ["Hello ", "there."]
    assert _turn_texts(captured) == ["Hello there."]
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert sum(len(f.audio) for f in audio) == 2 * len(AUDIO_CHUNK)
    assert len({f.context_id for f in audio}) == 1
    assert any(isinstance(f, TTSStoppedFrame) for f in down)
    assert not any(isinstance(f, ErrorFrame) for f in up)


async def test_speech_inside_a_reply_waits_for_the_turn_before_it():
    """Deepdub merges text sent before a turn's isFinal, so turns go out one at a time."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        down, up = await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame(text="Before "),
                TTSSpeakFrame(text="Filler."),
                LLMTextFrame(text="after."),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.5),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert _turn_texts(captured) == ["Before ", "Filler.", "after."]
    assert captured.get("violations", 0) == 0
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    by_context: dict[str, int] = {}
    for f in audio:
        by_context[f.context_id] = by_context.get(f.context_id, 0) + len(f.audio)
    # The reply is one context spoken as two turns, around the filler's own context.
    assert sorted(by_context.values()) == [2 * len(AUDIO_CHUNK), 4 * len(AUDIO_CHUNK)]
    stops = [f.context_id for f in down if isinstance(f, TTSStoppedFrame)]
    assert sorted(stops) == sorted(by_context)
    assert not any(isinstance(f, ErrorFrame) for f in up)


async def test_interruption_cancels_and_keeps_the_connection():
    captured: dict = {"messages": []}
    async with serve(_server(captured, hold_first_turn=True), "127.0.0.1", 0) as server:
        service = _service(server)
        down, _ = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="This reply gets cut off."),
                SleepFrame(sleep=0.2),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="Go ahead."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert len(_sent(captured, "cancel")) == 1
    assert captured["connections"] == 1
    assert _turn_texts(captured)[-1] == "Go ahead."
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert sum(len(f.audio) for f in audio) >= 2 * len(AUDIO_CHUNK)
    assert not service.get_audio_contexts()


async def test_interruption_after_the_turn_finished_sends_no_cancel():
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from Deepdub."),
                SleepFrame(sleep=0.3),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
            ],
        )

    assert not _sent(captured, "cancel")


async def test_session_error_is_reported_and_the_turn_cancelled():
    captured: dict = {"messages": []}
    error = {"action": "error", "message": "queue route not found for model: x", "time": 1}
    async with serve(_server(captured, error=error), "127.0.0.1", 0) as server:
        _, up = await run_test(
            _service(server),
            frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.3)],
        )

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert any("queue route not found" in f.error for f in errors)
    assert len(_sent(captured, "cancel")) == 1


async def test_a_turn_that_never_finishes_times_out():
    captured: dict = {"messages": []}
    async with serve(_server(captured, mute=True), "127.0.0.1", 0) as server:
        _, up = await run_test(
            _service(server, turn_timeout=0.3),
            frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=1.0)],
        )

    assert any("no audio" in f.error for f in up if isinstance(f, ErrorFrame))
    assert len(_sent(captured, "cancel")) == 1


async def test_settings_update_reconfigures_without_reconnecting():
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                TTSUpdateSettingsFrame(
                    delta=DeepdubTTSService.Settings(voice="voice-2", target_gender="female")
                ),
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="New voice."),
                SleepFrame(sleep=0.3),
            ],
        )

    configs = _sent(captured, "stream-config")
    assert captured["connections"] == 1
    assert len(configs) == 2
    assert configs[1]["data"]["voicePromptId"] == "voice-2"
    assert configs[1]["data"]["targetGender"] == "female"
    assert captured["messages"].index(configs[1]) < captured["messages"].index(
        _sent(captured, "stream-text")[0]
    )


@pytest.mark.parametrize(
    ("language", "locale"),
    [
        (Language.EN_US, "en-US"),
        (Language.EN, "en-US"),
        (Language.HE, "he-IL"),
        (Language.ES_MX, "es-MX"),
        (Language.PT, "pt-BR"),
        (Language.SW, None),
    ],
)
def test_language_to_locale(language, locale):
    assert DeepdubTTSService(api_key="dd-test").language_to_service_language(language) == locale


@pytest.mark.parametrize(("rate", "stream_rate"), [(16000, 16000), (12000, 16000), (96000, 48000)])
def test_stream_sample_rate(rate, stream_rate):
    service = DeepdubTTSService(api_key="dd-test")
    service._sample_rate = rate
    assert service.stream_sample_rate == stream_rate

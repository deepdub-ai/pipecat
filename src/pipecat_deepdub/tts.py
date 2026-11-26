#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Deepdub streaming text-to-speech service.

Speaks over Deepdub's streaming endpoint (``wss://wss.deepdub.ai/ws``), where text
streams in and audio streams out over one long-lived connection:

- ``stream-config`` configures the connection: model, voice, locale and audio format.
  It is sent once on connect, and again when settings change.
- ``stream-text`` sends text as the LLM produces it. Deepdub buffers it and picks
  segment boundaries itself, so tokens can be forwarded as they arrive.
- ``end-stream`` closes a turn (one reply). Its last audio chunk carries ``isFinal``.
- ``cancel`` aborts the turn in flight. Deepdub confirms with a ``cancelled`` status.

Deepdub speaks one turn at a time: text sent before a turn's ``isFinal`` joins that
turn. A turn that starts while another is still being spoken is therefore held here
and sent once the earlier turn has finished.

See https://docs.deepdub.ai/api-reference/websocket/streaming.
"""

import asyncio
import base64
import json
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven
from websockets.protocol import State

from pipecat_deepdub.models import (
    DEFAULT_LOCALE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_URL,
    DEFAULT_VOICE_PROMPT_ID,
    TTS_SAMPLE_RATES,
    locale_for,
)

_KEEPALIVE_INTERVAL = 20.0
_DEFAULT_TURN_TIMEOUT = 10.0


@dataclass
class DeepdubTTSSettings(TTSSettings):
    """Settings for DeepdubTTSService.

    ``model`` is a Deepdub model such as ``"dd-etts-4.0"``, ``voice`` a voice prompt
    ID, and ``language`` the locale to speak, such as ``"en-US"`` or ``"he-IL"``.

    Parameters:
        target_gender: The speaker's gender, ``"male"`` or ``"female"``, used for
            language-specific handling such as Hebrew diacritics. ``None`` leaves it
            to Deepdub.
        accept_emojis: Whether flag and emotion emojis in the text switch locale or
            emotion instead of being spoken. ``None`` leaves Deepdub's default (off).
    """

    target_gender: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    accept_emojis: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _Turn:
    """A Pipecat audio context, spoken by Deepdub as one turn."""

    context_id: str
    # Text not yet sent, because the turn is waiting for an earlier one to finish.
    pending: list[str] = field(default_factory=list)
    has_speech: bool = False
    ended: bool = False
    started: bool = False
    end_sent_at: float | None = None


class DeepdubTTSService(WebsocketTTSService):
    """Deepdub streaming text-to-speech over a persistent WebSocket.

    Tokens are forwarded as they arrive (``TextAggregationMode.TOKEN`` by default),
    each reply is closed with ``end-stream`` when the LLM finishes it, and an
    interrupted reply is cancelled without reconnecting.

    Example::

        tts = DeepdubTTSService(
            api_key=os.environ["DEEPDUB_API_KEY"],
            settings=DeepdubTTSService.Settings(
                voice="b2abb241-ac92-48bf-a890-fec03f43e209",
                language="en-US",
            ),
        )
    """

    Settings = DeepdubTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_TTS_URL,
        sample_rate: int | None = None,
        enable_logging: bool | None = None,
        turn_timeout: float = _DEFAULT_TURN_TIMEOUT,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepdub TTS service.

        Args:
            api_key: Deepdub API key.
            url: Streaming endpoint. Use ``wss://wss.eu.deepdub.ai/ws`` to process
                text and audio in the EU.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline's
                output rate. Deepdub streams 8000, 16000, 22050, 24000, 32000, 36000,
                44100 or 48000 Hz; any other rate is resampled from the next one up.
            enable_logging: Whether Deepdub may keep this session's text in its
                server-side logs. Set to False for confidential text. None leaves
                Deepdub's default (on).
            turn_timeout: Seconds to wait for audio after a reply is complete before
                giving up on it. Each chunk Deepdub sends restarts the wait.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        default_settings = self.Settings(
            model=DEFAULT_TTS_MODEL,
            voice=DEFAULT_VOICE_PROMPT_ID,
            language=DEFAULT_LOCALE,
            target_gender=None,
            accept_emojis=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        kwargs.setdefault("text_aggregation_mode", TextAggregationMode.TOKEN)
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=False,
            pause_frame_processing=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._enable_logging = enable_logging
        self._turn_timeout = turn_timeout

        self._receive_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        # Turns in the order Deepdub speaks them. Only the first is on the wire.
        self._turns: deque[_Turn] = deque()
        # Contexts Pipecat has flushed, so their audio ends with their last turn.
        self._flushed_contexts: set[str] = set()
        # Set from sending cancel until Deepdub confirms it; audio in between is stale.
        self._cancelling = False
        self._last_message_at = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepdub TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language to a Deepdub locale.

        Args:
            language: The language to convert.

        Returns:
            The Deepdub locale, such as ``"he-IL"``, or None if it isn't supported.
        """
        return locale_for(str(language.value))

    @property
    def stream_sample_rate(self) -> int:
        """The rate Deepdub streams at: the output rate, or the next supported one up."""
        for rate in TTS_SAMPLE_RATES:
            if rate >= self.sample_rate:
                return rate
        return TTS_SAMPLE_RATES[-1]

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Connecting here keeps the connection setup off the first reply's latency.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Deepdub applies a new ``stream-config`` to text sent after it, so changes take
        effect without reconnecting.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed and self._is_open():
            try:
                await self._send(self._config_message())
            except Exception as e:
                logger.warning(f"{self}: error sending stream-config: {e}")
        return changed

    #
    # Connection
    #

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        if not self._watchdog_task:
            self._watchdog_task = self.create_task(self._watchdog_handler(), name="turn_watchdog")
        self._create_keepalive_task(_KEEPALIVE_INTERVAL)

    async def _disconnect(self):
        await super()._disconnect()
        await self._cancel_keepalive_task()
        if self._watchdog_task:
            await self.cancel_task(self._watchdog_task)
            self._watchdog_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._is_open():
                return
            logger.debug(f"{self}: connecting to {self._url}")
            self._websocket = await self._websocket_connect(
                self._url, additional_headers={"x-api-key": self._api_key}
            )
            self._reset_turns()
            await self._send(self._config_message())
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Error connecting to Deepdub TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Deepdub TTS")
                await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self}: error closing websocket: {e}")
        finally:
            self._websocket = None
            self._reset_turns()
            await self._call_event_handler("on_disconnected")

    async def _close_connection_contexts(self):
        # Turns in flight don't survive the connection; the base class ends their
        # audio contexts.
        self._reset_turns()
        await super()._close_connection_contexts()

    def _reset_turns(self):
        self._turns.clear()
        self._flushed_contexts.clear()
        self._cancelling = False

    def _is_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send(self, message: dict[str, Any]):
        await self._get_websocket().send(json.dumps(message, ensure_ascii=False))

    async def _send_keepalive(self):
        await self._send({"action": "ping"})

    def _config_message(self) -> dict[str, Any]:
        s = self._settings
        data: dict[str, Any] = {
            "model": s.model,
            "voicePromptId": s.voice,
            "locale": s.language,
            "format": "s16le",
            "sampleRate": self.stream_sample_rate,
        }
        if self._enable_logging is not None:
            data["enableLogging"] = self._enable_logging
        if s.target_gender:
            data["targetGender"] = s.target_gender
        if s.accept_emojis is not None:
            data["acceptEmojis"] = s.accept_emojis
        return {"action": "stream-config", "data": data}

    #
    # Turns
    #

    def _open_turn(self, context_id: str) -> _Turn | None:
        # A context can own several turns: a TTSSpeakFrame mid-reply splits the
        # reply's text around it. Only its latest turn can still take text.
        for turn in reversed(self._turns):
            if turn.context_id == context_id:
                return None if turn.ended else turn
        return None

    async def _send_turns(self):
        """Send what the first turn has waiting, and finish turns with nothing to say."""
        while self._turns and not self._cancelling:
            turn = self._turns[0]
            if not turn.has_speech:
                if not turn.ended:
                    return
                # A turn of only whitespace would never get an isFinal from Deepdub.
                self._turns.popleft()
                await self._maybe_finish_context(turn.context_id)
                continue
            turn.started = True
            for text in turn.pending:
                await self._send({"action": "stream-text", "data": {"text": text}})
            turn.pending.clear()
            if turn.ended and turn.end_sent_at is None:
                await self._send({"action": "end-stream"})
                turn.end_sent_at = time.monotonic()
            return

    async def _end_turn(self, turn: _Turn):
        if not turn.ended:
            turn.ended = True
            await self._send_turns()

    async def _maybe_finish_context(self, context_id: str):
        """End a context's audio once it is flushed and all its turns are spoken."""
        if context_id not in self._flushed_contexts:
            return
        if any(turn.context_id == context_id for turn in self._turns):
            return
        await self._finish_context(context_id)

    async def _finish_context(self, context_id: str):
        self._flushed_contexts.discard(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _abort_current_turn(self):
        """Cancel the turn on the wire and end its context, with any turns it still has."""
        if not self._turns or not self._turns[0].started:
            return
        context_id = self._turns[0].context_id
        self._turns = deque(turn for turn in self._turns if turn.context_id != context_id)
        await self._send_cancel()
        await self._finish_context(context_id)

    async def _send_cancel(self):
        self._cancelling = True
        if self._is_open():
            try:
                await self._send({"action": "cancel"})
            except Exception as e:
                logger.warning(f"{self}: error sending cancel: {e}")

    async def flush_audio(self, context_id: str | None = None):
        """Close the reply: Deepdub speaks what it has buffered and ends the turn.

        Args:
            context_id: The audio context to close. If None, the latest turn.
        """
        if context_id is None:
            if not self._turns:
                return
            context_id = self._turns[-1].context_id
        self._flushed_contexts.add(context_id)
        if turn := self._open_turn(context_id):
            await self._end_turn(turn)
        else:
            await self._maybe_finish_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel an interrupted reply, keeping the connection for the next one.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        interrupted = [turn for turn in self._turns if turn.context_id == context_id]
        for turn in interrupted:
            self._turns.remove(turn)
        self._flushed_contexts.discard(context_id)
        if any(turn.started for turn in interrupted):
            await self._send_cancel()
        await super().on_audio_context_interrupted(context_id)

    #
    # Server messages
    #

    async def _receive_messages(self):
        async for message in self._get_websocket():
            self._last_message_at = time.monotonic()
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: ignoring a message that isn't JSON")
                continue
            action = msg.get("action")
            if action == "error":
                await self._handle_error(msg.get("message", ""), session=True)
            elif action == "status":
                if msg.get("message") == "cancelled":
                    self._cancelling = False
                    await self._send_turns()
            elif action is None:
                if msg.get("error"):
                    await self._handle_error(msg["error"], session=False)
                else:
                    await self._handle_audio(msg)

    async def _handle_audio(self, msg: dict[str, Any]):
        if self._cancelling or msg.get("isCancelled"):
            return
        if not self._turns or not self._turns[0].started:
            return
        turn = self._turns[0]
        if data := msg.get("data"):
            frame = TTSAudioRawFrame(
                audio=base64.b64decode(data),
                sample_rate=self.stream_sample_rate,
                num_channels=1,
                context_id=turn.context_id,
            )
            await self.append_to_audio_context(turn.context_id, frame)
        if msg.get("isFinal"):
            self._turns.popleft()
            await self._maybe_finish_context(turn.context_id)
            await self._send_turns()

    async def _handle_error(self, message: str, *, session: bool):
        # A session error (a bad model, say) is in the configuration, so it would
        # recur on every turn; a generation error concerns this turn's text.
        category = ErrorCategory.INVALID_REQUEST if session else ErrorCategory.SERVER
        await self.push_error(error_msg=f"Deepdub TTS error: {message}", category=category)
        await self._abort_current_turn()

    async def _watchdog_handler(self):
        """End a reply whose audio stopped arriving after it was complete."""
        while True:
            await asyncio.sleep(min(1.0, self._turn_timeout / 4))
            turn = self._turns[0] if self._turns else None
            if turn is None or turn.end_sent_at is None or self._cancelling:
                continue
            silent_for = time.monotonic() - max(turn.end_sent_at, self._last_message_at)
            if silent_for >= self._turn_timeout:
                await self.push_error(
                    error_msg=f"Deepdub TTS sent no audio for {self._turn_timeout}s",
                    category=ErrorCategory.SERVER,
                )
                await self._abort_current_turn()

    #
    # Synthesis
    #

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to Deepdub. Audio arrives on the receive task.

        Args:
            text: The text to synthesize.
            context_id: The audio context the text belongs to.

        Yields:
            Frame: None, as audio arrives via the WebSocket receive task.
        """
        try:
            if not self._is_open():
                await self._connect()

            turn = self._turns[-1] if self._turns else None
            if turn is None or turn.context_id != context_id or turn.ended:
                if turn is not None and not turn.ended:
                    # A separate utterance (a TTSSpeakFrame mid-reply) must not join
                    # the open turn, so that turn is closed first.
                    await self._end_turn(turn)
                turn = _Turn(context_id=context_id)
                self._turns.append(turn)

            turn.pending.append(text)
            turn.has_speech = turn.has_speech or bool(text.strip())
            await self._send_turns()
            await self.start_tts_usage_metrics(text)
        except Exception as e:
            if failed := self._open_turn(context_id):
                self._turns.remove(failed)
            yield ErrorFrame(error=f"Deepdub TTS error: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            return

        yield None

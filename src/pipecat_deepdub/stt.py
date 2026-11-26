#
# Copyright (c) 2025-2026, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Deepdub real-time speech-to-text service.

Streams audio to ``wss://wsgw.deepdub.ai/transcribe-websocket`` and turns Deepdub's
tokens into Pipecat transcription frames. Deepdub recognizes Hebrew and English, and
can tell which one is spoken in each utterance.

- The first message configures the session; audio follows as binary PCM frames.
- Tokens arrive as they are recognized, provisional at first and then final.
- An ``<end>`` token marks an endpoint (the speaker finished), and ``<fin>`` answers
  a ``finalize`` request.
- An empty text message ends the audio, and Deepdub answers with ``finished``.
"""

import json
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given
from websockets.protocol import State

from pipecat_deepdub.models import DEFAULT_STT_MODEL, DEFAULT_STT_URL, stt_language_for

_END_TOKEN = "<end>"
_FINALIZED_TOKEN = "<fin>"
_KEEPALIVE_MESSAGE = json.dumps({"type": "keepalive"})
_FINALIZE_MESSAGE = json.dumps({"type": "finalize"})

DEEPDUB_TTFS_P99 = 0.35
"""P99 seconds from the end of speech to the final transcript, after a finalize."""


@dataclass
class DeepdubSTTSettings(STTSettings):
    """Settings for DeepdubSTTService.

    ``language`` (``"en"`` or ``"he"``) tells Deepdub which language to expect. Leave
    it unset to let Deepdub detect Hebrew or English in each utterance.

    Parameters:
        language_hints_strict: Whether to recognize only the hinted language, rather
            than treat it as a preference.
        context_terms: Words and names to recognize reliably, such as product names.
        context_text: Background text about the conversation, to improve accuracy.
        max_endpoint_delay_ms: Longest wait, in milliseconds, after speech stops
            before Deepdub reports an endpoint (500 to 3000).
        client_reference_id: Your own identifier for the session, for support.
    """

    language_hints_strict: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context_terms: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context_text: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_endpoint_delay_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    client_reference_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class DeepdubSTTService(WebsocketSTTService):
    """Deepdub real-time speech-to-text for Hebrew and English.

    Interim transcripts stream while the user speaks. A final transcript is pushed at
    each endpoint, or as soon as the local VAD reports the user stopped speaking, which
    sends Deepdub a ``finalize``.

    Example::

        stt = DeepdubSTTService(
            api_key=os.environ["DEEPDUB_API_KEY"],
            settings=DeepdubSTTService.Settings(language="he"),
        )
    """

    Settings = DeepdubSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_STT_URL,
        sample_rate: int | None = None,
        vad_force_turn_endpoint: bool = True,
        should_interrupt: bool = True,
        ttfs_p99_latency: float | None = DEEPDUB_TTFS_P99,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepdub STT service.

        Args:
            api_key: Deepdub API key.
            url: Speech-to-text endpoint.
            sample_rate: Input sample rate in Hz. If None, uses the pipeline's input rate.
            vad_force_turn_endpoint: Turn detection mode. When True (the default),
                Pipecat's turn strategies decide when the user's turn ends, and a VAD
                stop sends ``finalize`` so the transcript arrives sooner. When False,
                Deepdub's endpoint detection ends the turn: the service proposes a turn
                start on the local VAD (or the first word) and a turn stop at each
                endpoint.
            should_interrupt: Whether the user speaking interrupts the bot, in Deepdub
                turn detection mode (``vad_force_turn_endpoint=False``).
            ttfs_p99_latency: P99 seconds from the end of speech to the final
                transcript. Override it for your deployment.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=DEFAULT_STT_MODEL,
            language=None,
            language_hints_strict=None,
            context_terms=None,
            context_text=None,
            max_endpoint_delay_ms=None,
            client_reference_id=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        _validate(default_settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=1,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._should_interrupt = should_interrupt

        self._receive_task = None
        self._final_tokens: list[dict[str, Any]] = []
        self._user_turn_open = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepdub STT supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language to a Deepdub speech-to-text language.

        Args:
            language: The language to convert.

        Returns:
            ``"en"`` or ``"he"``, or None if Deepdub doesn't recognize the language.
        """
        return stt_language_for(str(language.value))

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies in Deepdub turn detection mode.

        Returns:
            The STT metadata frame, recommending that the user aggregator follow this
            service's turn proposals when ``vad_force_turn_endpoint`` is False.
        """
        frame = super().service_metadata_frame()
        if not self._vad_force_turn_endpoint:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
        return frame

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting so the new session configuration applies.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            _validate(self._settings)
            await self._request_reconnect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """End the audio, then close the connection.

        Args:
            frame: The end frame.
        """
        if self._is_open():
            try:
                await self._get_websocket().send("")
            except Exception as e:
                logger.warning(f"{self}: end-of-audio send failed: {e}")
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Close the connection without waiting for the remaining transcript.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Deepdub. Transcripts arrive on the receive task.

        Args:
            audio: Raw 16-bit mono PCM audio.

        Yields:
            Frame: None, as transcripts arrive via the WebSocket receive task.
        """
        if self._is_open():
            try:
                await self._get_websocket().send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, finalizing the transcript when the user stops speaking.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._user_turn_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._vad_force_turn_endpoint:
            if self._is_open():
                try:
                    await self._get_websocket().send(_FINALIZE_MESSAGE)
                    self.request_finalize()
                except Exception as e:
                    logger.warning(f"{self}: finalize send failed: {e}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a transcription for tracing."""
        pass

    #
    # Turns (Deepdub turn detection mode only)
    #

    async def _user_turn_started(self):
        if self._vad_force_turn_endpoint or self._user_turn_open:
            return
        self._user_turn_open = True
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _user_turn_stopped(self):
        if self._vad_force_turn_endpoint or not self._user_turn_open:
            return
        self._user_turn_open = False
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    #
    # Connection
    #

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        # Started even when the connection failed: with no socket the receive loop
        # goes straight to its reconnect path, which retries the connect.
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
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
                self._url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            self._final_tokens = []
            await self._websocket.send(json.dumps(self._config_message()))
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Deepdub STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Deepdub STT")
                await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self}: error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _is_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send_keepalive(self, silence: bytes):
        """Send Deepdub's keepalive message instead of silent audio.

        Args:
            silence: Silent PCM audio (unused).
        """
        if self._websocket is not None:
            await self._websocket.send(_KEEPALIVE_MESSAGE)

    def _config_message(self) -> dict[str, Any]:
        s = self._settings
        config: dict[str, Any] = {
            "model": s.model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "enable_endpoint_detection": True,
        }
        if s.language:
            config["language_hints"] = [s.language]
            if s.language_hints_strict is not None:
                config["language_hints_strict"] = s.language_hints_strict
        if s.max_endpoint_delay_ms is not None:
            config["max_endpoint_delay_ms"] = s.max_endpoint_delay_ms
        context: dict[str, Any] = {}
        if s.context_terms:
            context["terms"] = list(s.context_terms)
        if s.context_text:
            context["text"] = s.context_text
        if context:
            config["context"] = context
        if s.client_reference_id:
            config["client_reference_id"] = s.client_reference_id
        return config

    #
    # Server messages
    #

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                content = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: ignoring a message that isn't JSON")
                continue

            if content.get("error_code") or content.get("error_message"):
                await self._end_utterance()
                await self.push_error(
                    error_msg=(
                        f"Deepdub STT error {content.get('error_code')}"
                        f" ({content.get('error_type')}): {content.get('error_message')}"
                    ),
                    category=_error_category(content.get("error_code")),
                )
                continue

            non_final: list[dict[str, Any]] = []
            for token in content.get("tokens") or []:
                text = token.get("text", "")
                is_marker = text in (_END_TOKEN, _FINALIZED_TOKEN)
                if not is_marker:
                    await self._user_turn_started()
                if token.get("is_final"):
                    if text == _FINALIZED_TOKEN:
                        self.confirm_finalize()
                        await self._end_utterance()
                    elif text == _END_TOKEN:
                        await self._end_utterance()
                    else:
                        self._final_tokens.append(token)
                elif not is_marker:
                    non_final.append(token)

            # Final words are held until the endpoint, so the transcript is pushed
            # whole; until then they show in the interim transcript.
            if self._final_tokens or non_final:
                tokens = self._final_tokens + non_final
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text="".join(t.get("text", "") for t in tokens),
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=_language_of(tokens),
                        result=tokens,
                    )
                )

            if content.get("finished"):
                await self._end_utterance()
                return

    async def _end_utterance(self):
        """Push the final transcript at an endpoint, and close the turn it ends."""
        if self._final_tokens:
            tokens, self._final_tokens = self._final_tokens, []
            text = "".join(t.get("text", "") for t in tokens).strip()
            if text:
                language = _language_of(tokens)
                # Usage goes out first so tracing attaches it to the span the
                # transcript closes.
                await self.emit_stt_usage_metrics()
                await self._push_transcription_runs(
                    [(None, text)], language=language, result=tokens, finalized=True
                )
                await self._handle_transcription(text, is_final=True, language=language)
        else:
            # A finalize that found nothing left to transcribe still confirms the
            # transcripts already pushed.
            await self._push_transcription_finalized()
        await self._user_turn_stopped()


def _language_of(tokens: list[dict[str, Any]]) -> Language | None:
    counts: Counter[str] = Counter()
    for token in tokens:
        if language := token.get("language"):
            counts[language] += len(token.get("text", ""))
    if not counts:
        return None
    try:
        return Language(counts.most_common(1)[0][0])
    except ValueError:
        return None


def _error_category(code: Any) -> ErrorCategory:
    if code in (401, 403):
        return ErrorCategory.AUTHENTICATION
    if code == 429:
        return ErrorCategory.RATE_LIMIT
    if isinstance(code, int) and 400 <= code < 500:
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.SERVER


def _validate(settings: DeepdubSTTSettings):
    language = settings.language
    code = str(language.value) if isinstance(language, Language) else str(language)
    if is_given(language) and language and stt_language_for(code) is None:
        raise ValueError(f"Deepdub speech-to-text recognizes English and Hebrew, got {language!r}")
    delay = settings.max_endpoint_delay_ms
    if is_given(delay) and delay is not None and not 500 <= delay <= 3000:
        raise ValueError(f"max_endpoint_delay_ms must be between 500 and 3000, got {delay}")

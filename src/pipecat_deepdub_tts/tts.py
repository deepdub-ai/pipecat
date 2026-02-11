#
# Copyright (c) 2025, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Deepdub AI text-to-speech service implementation.

This module provides a WebSocket-based streaming TTS service using Deepdub AI's API.
It uses the ``deepdub`` package's ``DeepdubClient`` for WebSocket communication and
extends Pipecat's ``InterruptibleTTSService`` for seamless pipeline integration.

The service connects to Deepdub's streaming WebSocket endpoint, sends text chunks
as they arrive from the LLM, and pushes raw PCM audio frames into the pipeline.

**Streaming Protocol:**

- ``stream-config``: Initial configuration with model, voice, format, and sample rate
- ``stream-text``: Text chunks sent for synthesis
- Audio responses arrive as JSON messages with base64-encoded ``s16le`` PCM data

Example::

    tts = DeepdubTTSService(
        api_key="your-api-key",
        voice_id="your-voice-prompt-id",
        model="dd-etts-2.5",
        params=DeepdubTTSService.InputParams(
            locale="en-US",
            temperature=0.7,
        ),
    )

See https://github.com/deepdub-ai/deepdub for full API details.
"""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from deepdub import DeepdubClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepdub TTS, you need to `pip install pipecat-deepdub-tts`."
    )
    raise Exception(f"Missing module: {e}")

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepdub TTS, you need to `pip install websockets`."
    )
    raise Exception(f"Missing module: {e}")


def language_to_deepdub_locale(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Deepdub locale codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Deepdub locale code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar-SA",
        Language.BG: "bg-BG",
        Language.CA: "ca-ES",
        Language.CS: "cs-CZ",
        Language.DA: "da-DK",
        Language.DE: "de-DE",
        Language.EL: "el-GR",
        Language.EN: "en-US",
        Language.ES: "es-ES",
        Language.FI: "fi-FI",
        Language.FR: "fr-FR",
        Language.HE: "he-IL",
        Language.HI: "hi-IN",
        Language.HR: "hr-HR",
        Language.HU: "hu-HU",
        Language.ID: "id-ID",
        Language.IT: "it-IT",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR",
        Language.MS: "ms-MY",
        Language.NL: "nl-NL",
        Language.NO: "nb-NO",
        Language.PL: "pl-PL",
        Language.PT: "pt-BR",
        Language.RO: "ro-RO",
        Language.RU: "ru-RU",
        Language.SK: "sk-SK",
        Language.SV: "sv-SE",
        Language.TH: "th-TH",
        Language.TR: "tr-TR",
        Language.UK: "uk-UA",
        Language.VI: "vi-VN",
        Language.ZH: "zh-CN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class DeepdubTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Deepdub AI.

    Provides streaming TTS with real-time audio generation using Deepdub's
    WebSocket streaming API. Uses the ``DeepdubClient`` from the ``deepdub``
    package for protocol communication.

    The service manages the WebSocket lifecycle (connect/disconnect) to fit
    Pipecat's pipeline model, while delegating protocol details to the client.

    Args:
        api_key: Deepdub API key for authentication.
        voice_id: Voice prompt ID to use for synthesis.
        model: TTS model name (e.g. "dd-etts-2.5", "dd-etts-3.0").
        sample_rate: Audio sample rate in Hz. Valid: 8000, 16000, 22050, 24000, 44100, 48000.
        params: Optional input parameters for voice customization.
        **kwargs: Additional arguments passed to InterruptibleTTSService.

    Example::

        tts = DeepdubTTSService(
            api_key=os.getenv("DEEPDUB_API_KEY"),
            voice_id=os.getenv("DEEPDUB_VOICE_ID"),
            model="dd-etts-2.5",
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Deepdub TTS configuration.

        Attributes:
            locale: Language locale for synthesis (e.g. "en-US"). Defaults to "en-US".
            temperature: Controls output variability. Higher values produce more
                varied speech. Defaults to None (server default).
            variance: Controls variance in the generated speech. Defaults to None.
            tempo: Speech tempo multiplier. Mutually exclusive with duration.
                Defaults to None (server default).
            prompt_boost: Whether to enable prompt boosting for improved quality.
                Defaults to None (server default).
            accent_base_locale: Base locale for accent control (e.g. "en-US").
                All three accent params must be provided together or none.
            accent_locale: Target accent locale (e.g. "fr-FR").
            accent_ratio: Accent blending ratio (0.0 to 1.0).
        """

        locale: str = Field(
            default="en-US",
            description="Language locale for synthesis.",
        )
        temperature: Optional[float] = Field(
            default=None,
            description="Controls output variability.",
        )
        variance: Optional[float] = Field(
            default=None,
            description="Controls variance in generated speech.",
        )
        tempo: Optional[float] = Field(
            default=None,
            description="Speech tempo multiplier.",
        )
        prompt_boost: Optional[bool] = Field(
            default=None,
            description="Enable prompt boosting for improved quality.",
        )
        accent_base_locale: Optional[str] = Field(
            default=None,
            description="Base locale for accent control. All three accent params required together.",
        )
        accent_locale: Optional[str] = Field(
            default=None,
            description="Target accent locale.",
        )
        accent_ratio: Optional[float] = Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Accent blending ratio (0.0 to 1.0).",
        )

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "dd-etts-2.5",
        sample_rate: int = 16000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Deepdub TTS service.

        Args:
            api_key: Deepdub API key for authentication.
            voice_id: Voice prompt ID for TTS synthesis.
            model: TTS model to use. Defaults to "dd-etts-2.5".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Optional input parameters for voice customization.
            **kwargs: Additional arguments passed to InterruptibleTTSService.

        Raises:
            ValueError: If api_key is empty or sample_rate is invalid.
        """
        if not api_key or not api_key.strip():
            raise ValueError("Deepdub API key is required and cannot be empty")

        valid_sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
        if sample_rate not in valid_sample_rates:
            raise ValueError(
                f"sample_rate must be one of {valid_sample_rates}, got {sample_rate}"
            )

        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or DeepdubTTSService.InputParams()

        self._client = DeepdubClient(api_key=api_key)
        self._voice_id = voice_id

        self._settings = {
            "model": model,
            "locale": params.locale,
            "voice_prompt_id": voice_id,
            "format": "s16le",
            "sample_rate": sample_rate,
            "temperature": params.temperature,
            "variance": params.variance,
            "tempo": params.tempo,
            "prompt_boost": params.prompt_boost,
            "accent_base_locale": params.accent_base_locale,
            "accent_locale": params.accent_locale,
            "accent_ratio": params.accent_ratio,
        }

        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._context_id: Optional[str] = None

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepdub service supports metrics generation.
        """
        return True

    def set_voice(self, voice_id: str) -> None:
        """Set the voice ID for TTS synthesis.

        Args:
            voice_id: The voice prompt identifier to use.
        """
        logger.info(f"Setting Deepdub TTS voice to: [{voice_id}]")
        self._voice_id = voice_id
        self._settings["voice_prompt_id"] = voice_id

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Deepdub locale format.

        Args:
            language: The language to convert.

        Returns:
            The Deepdub-specific locale code, or None if not supported.
        """
        return language_to_deepdub_locale(language)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and flush audio at the end of a full LLM response.

        Args:
            frame: The frame to process.
            direction: The direction to process the frame.
        """
        await super().process_frame(frame, direction)

        # When the LLM finishes responding, flush any remaining text
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def start(self, frame: StartFrame):
        """Start the Deepdub TTS service.

        Initializes the WebSocket connection and starts the background
        receive task for processing audio chunks.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepdub TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepdub TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis.

        Deepdub's streaming protocol processes text as it arrives and sends
        audio back immediately. No explicit flush signal is needed.
        """
        logger.debug(f"{self}: flush_audio (no-op for Deepdub streaming)")

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice changed.

        Args:
            settings: Dictionary of settings to update.
        """
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if prev_voice != self._voice_id:
            self._settings["voice_prompt_id"] = self._voice_id
            await self._disconnect()
            await self._connect()

    async def _connect(self):
        """Connect to Deepdub WebSocket and start background tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_handler())

    async def _disconnect(self):
        """Disconnect from Deepdub WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Deepdub streaming API.

        Opens a websocket to the Deepdub streaming endpoint using the client's
        configured URL, then sends the stream configuration message via the
        client's ``async_stream_config`` method.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            headers = {"x-api-key": self._client.api_key}
            url = self._client.base_websocket_streaming_url

            logger.debug(f"Connecting to Deepdub TTS at {url}")
            ws = await websocket_connect(url, additional_headers=headers)
            self._websocket = ws
            self._client.websocket = ws

            # Build accent control kwargs
            accent_kwargs = {}
            if (
                self._settings["accent_base_locale"] is not None
                and self._settings["accent_locale"] is not None
                and self._settings["accent_ratio"] is not None
            ):
                accent_kwargs = {
                    "accent_base_locale": self._settings["accent_base_locale"],
                    "accent_locale": self._settings["accent_locale"],
                    "accent_ratio": self._settings["accent_ratio"],
                }

            # Send stream configuration via the client
            await self._client.async_stream_config(
                model=self._settings["model"],
                locale=self._settings["locale"],
                voice_prompt_id=self._settings["voice_prompt_id"],
                format=self._settings["format"],
                sample_rate=self._settings["sample_rate"],
                temperature=self._settings["temperature"],
                variance=self._settings["variance"],
                tempo=self._settings["tempo"],
                prompt_boost=self._settings["prompt_boost"],
                **accent_kwargs,
            )

            logger.debug("Connected to Deepdub TTS WebSocket")

            await self._call_event_handler("on_connected")

        except Exception as e:
            logger.error(f"{self} connection error: {e}")
            self._websocket = None
            self._client.websocket = None
            await self.push_error(
                error_msg=f"{self} connection error: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Deepdub TTS")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._client.websocket = None
            self._context_id = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the active WebSocket connection.

        Returns:
            The active websocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Deepdub WebSocket not connected")

    async def _keepalive_handler(self):
        """Send periodic pings to keep the WebSocket connection alive."""
        while True:
            try:
                await asyncio.sleep(20)
                if self._websocket:
                    await self._websocket.ping()
                    logger.trace(f"{self}: keepalive ping sent")
            except Exception as e:
                logger.warning(f"{self}: keepalive ping failed: {e}")
                break

    async def _receive_messages(self):
        """Receive and process audio messages from Deepdub WebSocket.

        Continuously reads messages from the websocket. Each message is
        expected to be a bytes payload containing JSON with a base64-encoded
        ``data`` field holding s16le PCM audio.

        Messages with an ``error`` field cause an error to be pushed to the
        pipeline. Messages with ``isFinished: true`` indicate the server has
        temporarily finished sending audio for the current text.
        """
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = json.loads(message)
                elif isinstance(message, str):
                    msg = json.loads(message)
                else:
                    logger.warning(
                        f"{self} received unexpected message type: {type(message)}"
                    )
                    continue

                if msg.get("error"):
                    error_text = msg["error"]
                    await self.push_error(
                        error_msg=f"Deepdub TTS error: {error_text}"
                    )
                    break

                if msg.get("data"):
                    await self.stop_ttfb_metrics()
                    audio_data = base64.b64decode(msg["data"])
                    frame = TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=self._context_id,
                    )
                    await self.push_frame(frame)

            except json.JSONDecodeError as e:
                logger.warning(f"{self} failed to parse message as JSON: {e}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")
                await self.push_error(
                    error_msg=f"{self} error processing message: {e}",
                    exception=e,
                )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepdub's streaming WebSocket API.

        Sends text to the Deepdub streaming endpoint via the client's
        ``async_stream_text`` method. Audio is received asynchronously by
        the background receive task and pushed into the pipeline.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: No frames are yielded directly; audio arrives via the
            background receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        self._context_id = context_id

        try:
            await self.start_ttfb_metrics()
            await self._client.async_stream_text(text)
            await self.start_tts_usage_metrics(text)
        except Exception as e:
            await self.push_error(
                error_msg=f"Error sending text to Deepdub: {e}", exception=e
            )

        # AsyncGenerator: yield nothing, audio arrives via _receive_messages
        return
        yield  # noqa: unreachable — makes this function a generator

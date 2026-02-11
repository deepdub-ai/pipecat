#
# Copyright (c) 2025, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Tests for DeepdubTTSService.

Unit tests run without API keys (mocked).
Integration tests require DEEPDUB_API_KEY and save audio output for manual verification.
"""

import asyncio
import os
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from pipecat_deepdub_tts.tts import DeepdubTTSService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "output"

HAVE_API_KEY = bool(os.getenv("DEEPDUB_API_KEY"))
VOICE_ID = os.getenv("DEEPDUB_VOICE_ID", "test-voice-id")
MODEL = os.getenv("DEEPDUB_MODEL", "dd-etts-2.5")


def _write_wav(path: Path, pcm_data: bytes, sample_rate: int = 16000, num_channels: int = 1, sample_width: int = 2):
    """Write raw PCM data to a WAV file with proper headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# Unit Tests (no API key needed)
# ---------------------------------------------------------------------------


class TestDeepdubTTSServiceConstruction:
    """Test DeepdubTTSService constructor and configuration."""

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_valid_construction(self, mock_client_cls):
        """Service should initialize with valid parameters."""
        service = DeepdubTTSService(
            api_key="test-api-key",
            voice_id="test-voice-id",
            model="dd-etts-2.5",
            sample_rate=16000,
        )
        assert service._settings["model"] == "dd-etts-2.5"
        assert service._settings["voice_prompt_id"] == "test-voice-id"
        assert service._settings["sample_rate"] == 16000
        assert service._settings["format"] == "s16le"
        assert service._settings["locale"] == "en-US"
        assert service._context_id is None
        assert service._keepalive_task is None
        mock_client_cls.assert_called_once_with(api_key="test-api-key")

    def test_empty_api_key_raises(self):
        """Service should raise ValueError for empty API key."""
        with pytest.raises(ValueError, match="API key is required"):
            DeepdubTTSService(
                api_key="",
                voice_id="test-voice-id",
            )

    def test_whitespace_api_key_raises(self):
        """Service should raise ValueError for whitespace-only API key."""
        with pytest.raises(ValueError, match="API key is required"):
            DeepdubTTSService(
                api_key="   ",
                voice_id="test-voice-id",
            )

    def test_invalid_sample_rate_raises(self):
        """Service should raise ValueError for invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate must be one of"):
            DeepdubTTSService(
                api_key="test-api-key",
                voice_id="test-voice-id",
                sample_rate=12345,
            )

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_valid_sample_rates(self, mock_client_cls):
        """Service should accept all valid sample rates."""
        for rate in [8000, 16000, 22050, 24000, 44100, 48000]:
            service = DeepdubTTSService(
                api_key="test-api-key",
                voice_id="test-voice-id",
                sample_rate=rate,
            )
            assert service._settings["sample_rate"] == rate

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_custom_params(self, mock_client_cls):
        """Service should store custom InputParams correctly."""
        params = DeepdubTTSService.InputParams(
            locale="fr-FR",
            temperature=0.8,
            variance=0.5,
            tempo=1.2,
            prompt_boost=True,
            accent_base_locale="en-US",
            accent_locale="fr-FR",
            accent_ratio=0.7,
        )
        service = DeepdubTTSService(
            api_key="test-api-key",
            voice_id="test-voice-id",
            params=params,
        )
        assert service._settings["locale"] == "fr-FR"
        assert service._settings["temperature"] == 0.8
        assert service._settings["variance"] == 0.5
        assert service._settings["tempo"] == 1.2
        assert service._settings["prompt_boost"] is True
        assert service._settings["accent_base_locale"] == "en-US"
        assert service._settings["accent_locale"] == "fr-FR"
        assert service._settings["accent_ratio"] == 0.7

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_default_params(self, mock_client_cls):
        """Service should use sensible defaults when no params given."""
        service = DeepdubTTSService(
            api_key="test-api-key",
            voice_id="test-voice-id",
        )
        assert service._settings["locale"] == "en-US"
        assert service._settings["temperature"] is None
        assert service._settings["variance"] is None
        assert service._settings["tempo"] is None
        assert service._settings["prompt_boost"] is None
        assert service._settings["accent_base_locale"] is None


class TestDeepdubTTSServiceMetrics:
    """Test metrics support."""

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_can_generate_metrics(self, mock_client_cls):
        """Service should report metrics capability."""
        service = DeepdubTTSService(
            api_key="test-api-key",
            voice_id="test-voice-id",
        )
        assert service.can_generate_metrics() is True


class TestInputParams:
    """Test InputParams validation."""

    def test_default_input_params(self):
        """InputParams should have sensible defaults."""
        params = DeepdubTTSService.InputParams()
        assert params.locale == "en-US"
        assert params.temperature is None
        assert params.variance is None
        assert params.tempo is None
        assert params.prompt_boost is None
        assert params.accent_base_locale is None
        assert params.accent_locale is None
        assert params.accent_ratio is None

    def test_accent_ratio_bounds(self):
        """InputParams should validate accent_ratio bounds."""
        # Valid
        params = DeepdubTTSService.InputParams(accent_ratio=0.0)
        assert params.accent_ratio == 0.0

        params = DeepdubTTSService.InputParams(accent_ratio=1.0)
        assert params.accent_ratio == 1.0

        # Invalid
        with pytest.raises(Exception):
            DeepdubTTSService.InputParams(accent_ratio=1.5)

        with pytest.raises(Exception):
            DeepdubTTSService.InputParams(accent_ratio=-0.1)


# ---------------------------------------------------------------------------
# Integration Test (requires DEEPDUB_API_KEY, saves audio to disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_API_KEY, reason="DEEPDUB_API_KEY not set")
class TestDeepdubTTSIntegration:
    """Integration tests that call the real Deepdub API and save audio output.

    These tests require DEEPDUB_API_KEY and DEEPDUB_VOICE_ID to be set
    as environment variables (e.g. via a .env file).

    Generated audio is saved to tests/output/ for manual listening verification.
    """

    @pytest.mark.asyncio
    async def test_streaming_tts_generates_audio(self):
        """Call Deepdub streaming TTS and save the resulting audio as a WAV file.

        This test:
        1. Opens a streaming WebSocket connection via DeepdubClient
        2. Sends a short test sentence
        3. Collects all audio chunks
        4. Saves the result to tests/output/test_output.wav
        """
        from deepdub import DeepdubClient

        client = DeepdubClient()
        sample_rate = 16000
        test_text = "Hello, this is a test of the Deepdub text to speech integration with Pipecat."

        audio_chunks = []

        async with client.async_stream_connect(
            model=MODEL,
            locale="en-US",
            voice_prompt_id=VOICE_ID,
            sample_rate=sample_rate,
            format="s16le",
        ) as conn:
            await conn.async_stream_text(text=test_text)

            # Collect audio chunks with a timeout
            while True:
                wait_task = asyncio.create_task(conn.async_stream_recv_audio())
                try:
                    chunk = await asyncio.wait_for(wait_task, timeout=5.0)
                    if chunk:
                        audio_chunks.append(chunk)
                except asyncio.TimeoutError:
                    break

        assert len(audio_chunks) > 0, "Expected at least one audio chunk from Deepdub"

        # Concatenate all chunks and write to WAV
        pcm_data = b"".join(audio_chunks)
        output_path = OUTPUT_DIR / "test_output.wav"
        _write_wav(output_path, pcm_data, sample_rate=sample_rate)

        assert output_path.exists(), f"WAV file was not created at {output_path}"
        assert output_path.stat().st_size > 44, "WAV file appears to be empty (header only)"

        print(f"\nAudio saved to: {output_path.resolve()}")
        print(f"Audio size: {len(pcm_data)} bytes ({len(pcm_data) / (sample_rate * 2):.2f} seconds)")

    @pytest.mark.asyncio
    async def test_non_streaming_tts_generates_audio(self):
        """Call Deepdub non-streaming WebSocket TTS and save the result.

        This test uses the non-streaming async_tts method for comparison.
        Audio is saved to tests/output/test_output_non_streaming.wav

        Note: The non-streaming API uses WAV format. We request format="wav"
        (with headers) and parse the first chunk's header to determine the
        actual sample rate, since the server may return a different rate than
        requested.
        """
        from deepdub import DeepdubClient

        client = DeepdubClient()
        test_text = "This is a non-streaming test of Deepdub text to speech."

        # Use format="wav" to get complete WAV chunks with headers,
        # so we can detect the actual sample rate from the header.
        raw_chunks = []

        async with client.async_connect() as conn:
            async for chunk in conn.async_tts(
                text=test_text,
                voice_prompt_id=VOICE_ID,
                model=MODEL,
                locale="en-US",
                format="wav",
                sample_rate=16000,
            ):
                raw_chunks.append(chunk)

        assert len(raw_chunks) > 0, "Expected at least one audio chunk"

        # Parse the WAV header from the first chunk to get actual sample rate.
        # Deepdub WAV header is 68 bytes (0x44): RIFF + fmt (40 bytes) + data header
        first = raw_chunks[0]
        assert first[:4] == b"RIFF", "Expected WAV data from non-streaming API"
        actual_sample_rate = struct.unpack_from("<I", first, 24)[0]
        bits_per_sample = struct.unpack_from("<H", first, 34)[0]
        sample_width = bits_per_sample // 8
        wav_header_len = 68  # Deepdub's dd_wav_header_len = 0x44

        # Strip the 68-byte WAV header from each chunk to get raw PCM
        pcm_data = b"".join(chunk[wav_header_len:] for chunk in raw_chunks)

        output_path = OUTPUT_DIR / "test_output_non_streaming.wav"
        _write_wav(output_path, pcm_data, sample_rate=actual_sample_rate, sample_width=sample_width)

        assert output_path.exists(), f"WAV file was not created at {output_path}"
        assert output_path.stat().st_size > 44, "WAV file appears to be empty"

        duration = len(pcm_data) / (actual_sample_rate * sample_width)
        print(f"\nAudio saved to: {output_path.resolve()}")
        print(f"Audio: {actual_sample_rate} Hz, {bits_per_sample}-bit, {len(pcm_data)} bytes ({duration:.2f} seconds)")


# ---------------------------------------------------------------------------
# Pipeline Integration Test (requires DEEPDUB_API_KEY + OPENAI_API_KEY)
# ---------------------------------------------------------------------------

HAVE_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


class SegmentedAudioCaptureSink:
    """Captures TTSAudioRawFrame audio segmented by context_id.

    Each call to ``run_tts`` produces frames with a unique ``context_id``.
    This sink groups audio by that id, so each TTS response becomes a
    separate segment. Call ``get_segments()`` after the pipeline runs.
    """

    def __init__(self):
        from pipecat.frames.frames import TTSAudioRawFrame
        from pipecat.processors.frame_processor import FrameProcessor

        self._segments_by_ctx: dict[str, bytearray] = {}
        self._ctx_order: list[str] = []
        outer = self

        class _Sink(FrameProcessor):
            async def process_frame(sink_self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TTSAudioRawFrame):
                    ctx = getattr(frame, "context_id", None) or "_default"
                    if ctx not in outer._segments_by_ctx:
                        outer._segments_by_ctx[ctx] = bytearray()
                        outer._ctx_order.append(ctx)
                    outer._segments_by_ctx[ctx].extend(frame.audio)
                await sink_self.push_frame(frame, direction)

        self._processor = _Sink()

    @property
    def processor(self):
        return self._processor

    def get_segments(self) -> list[bytes]:
        """Return audio segments in the order they were first seen."""
        return [bytes(self._segments_by_ctx[ctx]) for ctx in self._ctx_order]

    def get_all_audio(self) -> bytes:
        """Return all captured audio concatenated."""
        return b"".join(self.get_segments())


@pytest.mark.skipif(
    not (HAVE_API_KEY and HAVE_OPENAI_KEY),
    reason="DEEPDUB_API_KEY and OPENAI_API_KEY required",
)
class TestDeepdubPipelineIntegration:
    """Full pipeline integration test: OpenAI LLM -> Deepdub TTS.

    Sends multiple user messages through a real pipeline, captures each
    TTS audio response separately, and saves them as individual WAV files
    for manual listening verification.

    Requires DEEPDUB_API_KEY, DEEPDUB_VOICE_ID, and OPENAI_API_KEY.
    """

    # Each tuple is (user_prompt, output_filename, description)
    INTERACTIONS = [
        (
            "Say hello in one short sentence.",
            "pipeline_01_greeting.wav",
            "Short greeting",
        ),
        (
            "Describe what a sunny day feels like in two sentences.",
            "pipeline_02_description.wav",
            "Medium-length description",
        ),
        (
            "Count from one to five, saying each number clearly.",
            "pipeline_03_counting.wav",
            "Structured counting",
        ),
    ]

    @pytest.mark.asyncio
    async def test_multi_turn_pipeline(self):
        """Multi-turn LLM + TTS pipeline producing separate audio files.

        The test:
        1. Builds an OpenAI LLM + Deepdub TTS pipeline
        2. Sends 3 different user prompts sequentially
        3. Captures each TTS response as a separate audio segment
        4. Saves each to tests/output/pipeline_01_*.wav, pipeline_02_*.wav, etc.
        """
        from pipecat.frames.frames import LLMRunFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )
        from pipecat.services.openai.llm import OpenAILLMService

        sample_rate = 16000

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        tts = DeepdubTTSService(
            api_key=os.getenv("DEEPDUB_API_KEY"),
            voice_id=VOICE_ID,
            model=MODEL,
            sample_rate=sample_rate,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Keep answers short, one to two "
                    "sentences maximum. Do not use special characters, bullet "
                    "points, or formatting that cannot be spoken aloud."
                ),
            },
        ]

        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

        audio_capture = SegmentedAudioCaptureSink()

        pipeline = Pipeline(
            [
                user_aggregator,
                llm,
                tts,
                audio_capture.processor,
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Queue all user messages with delays so each TTS response
        # completes before the next prompt is sent.
        async def send_prompts():
            for i, (prompt, _, desc) in enumerate(self.INTERACTIONS):
                if i > 0:
                    # Wait for the previous TTS response to finish
                    await asyncio.sleep(8)
                print(f"\n--- Sending prompt {i + 1}: {desc} ---")
                messages.append({"role": "user", "content": prompt})
                await task.queue_frames([LLMRunFrame()])
            # Wait for the last response, then cancel the pipeline
            await asyncio.sleep(10)
            await task.cancel()

        prompt_task = asyncio.create_task(send_prompts())

        runner = PipelineRunner()
        await runner.run(task)

        prompt_task.cancel()

        # Collect and save segments
        segments = audio_capture.get_segments()

        print(f"\n{'=' * 60}")
        print(f"Captured {len(segments)} audio segment(s)")
        print(f"{'=' * 60}")

        assert len(segments) > 0, "Pipeline produced no audio segments"

        for i, (prompt, filename, desc) in enumerate(self.INTERACTIONS):
            if i < len(segments):
                pcm_data = segments[i]
                output_path = OUTPUT_DIR / filename
                _write_wav(output_path, pcm_data, sample_rate=sample_rate)

                duration = len(pcm_data) / (sample_rate * 2)
                assert output_path.exists(), f"WAV file was not created: {output_path}"
                assert output_path.stat().st_size > 44, f"WAV file is empty: {filename}"

                print(f"  [{i + 1}] {desc}: {duration:.2f}s -> {filename}")
            else:
                print(f"  [{i + 1}] {desc}: NO AUDIO CAPTURED (segment missing)")

        # At least the first interaction must produce audio
        assert len(segments[0]) > 0, "First interaction produced no audio"
        print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")

#
# Copyright (c) 2025, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Tests for the deepdub_tts_basic example.

Validates that the example code is correct: imports resolve, services can be
constructed, pipeline assembles, and the bot entry-point functions exist with
the right signatures.

All tests run without API keys (mocked).

Note: Some pipecat transports (Daily, Twilio/FastAPI) require optional
dependencies that may not be installed in the dev environment. Tests that
depend on these are skipped when the deps are missing.
"""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "foundational"
EXAMPLE_FILE = EXAMPLE_DIR / "deepdub_tts_basic.py"


def _check_optional_import(module_path: str) -> bool:
    """Check if an optional module can be imported."""
    try:
        importlib.import_module(module_path)
        return True
    except (ModuleNotFoundError, Exception):
        return False


HAVE_FASTAPI = _check_optional_import("fastapi")
HAVE_DAILY = _check_optional_import("daily")

# All transport-related optional deps available
HAVE_TRANSPORT_DEPS = HAVE_FASTAPI and HAVE_DAILY


def _import_example_module():
    """Import the example module dynamically so we can inspect it."""
    spec = importlib.util.spec_from_file_location("deepdub_tts_basic", EXAMPLE_FILE)
    module = importlib.util.module_from_spec(spec)
    return spec, module


def _mock_missing_transport_modules():
    """Create mock modules for optional transport deps that aren't installed.

    Returns a dict suitable for use with patch.dict(sys.modules, ...).
    """
    mocks = {}

    if not HAVE_DAILY:
        # Mock the daily module and pipecat's Daily transport
        daily_mock = MagicMock()
        mocks["daily"] = daily_mock

        # Create a mock DailyParams class that inherits behavior
        daily_transport_mock = MagicMock()
        daily_transport_mock.DailyParams = type("DailyParams", (), {
            "__init__": lambda self, **kw: None,
            "audio_in_enabled": True,
            "audio_out_enabled": True,
        })
        mocks["pipecat.transports.daily"] = MagicMock()
        mocks["pipecat.transports.daily.transport"] = daily_transport_mock

    if not HAVE_FASTAPI:
        # Mock fastapi and pipecat's runner.types
        mocks["fastapi"] = MagicMock()

        runner_types_mock = MagicMock()
        runner_types_mock.RunnerArguments = type("RunnerArguments", (), {
            "handle_sigint": True,
        })
        mocks["pipecat.runner.types"] = runner_types_mock
        mocks["pipecat.runner.utils"] = MagicMock()
        mocks["pipecat.runner.run"] = MagicMock()

        ws_transport_mock = MagicMock()
        ws_transport_mock.FastAPIWebsocketParams = type("FastAPIWebsocketParams", (), {
            "__init__": lambda self, **kw: None,
            "audio_in_enabled": True,
            "audio_out_enabled": True,
        })
        mocks["pipecat.transports.websocket"] = MagicMock()
        mocks["pipecat.transports.websocket.fastapi"] = ws_transport_mock

    return mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExampleFileExists:
    """Ensure the example file is present and well-formed."""

    def test_example_file_exists(self):
        """The example script must exist on disk."""
        assert EXAMPLE_FILE.exists(), f"Example not found at {EXAMPLE_FILE}"

    def test_example_file_is_not_empty(self):
        """The example script must contain code."""
        assert EXAMPLE_FILE.stat().st_size > 0, "Example file is empty"

    def test_example_compiles(self):
        """The example script must be valid Python (no syntax errors)."""
        source = EXAMPLE_FILE.read_text(encoding="utf-8")
        compile(source, str(EXAMPLE_FILE), "exec")


class TestExampleImports:
    """Verify that all imports used by the example resolve correctly."""

    def test_pipecat_core_imports(self):
        """Core pipecat modules used by the example must be importable."""
        from pipecat.frames.frames import LLMRunFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.transports.base_transport import BaseTransport, TransportParams

    @pytest.mark.skipif(not HAVE_FASTAPI, reason="fastapi not installed")
    def test_pipecat_runner_imports(self):
        """Pipecat runner utilities used by the example must be importable."""
        from pipecat.runner.types import RunnerArguments
        from pipecat.runner.utils import create_transport

    def test_deepdub_tts_import(self):
        """The DeepdubTTSService must be importable from the package."""
        from pipecat_deepdub_tts import DeepdubTTSService
        assert DeepdubTTSService is not None

    @pytest.mark.skipif(not HAVE_DAILY, reason="daily not installed")
    def test_daily_transport_import(self):
        """DailyParams used in the example must be importable."""
        from pipecat.transports.daily.transport import DailyParams

    @pytest.mark.skipif(not HAVE_FASTAPI, reason="fastapi not installed")
    def test_fastapi_transport_import(self):
        """FastAPIWebsocketParams used in the example must be importable."""
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    def test_vad_import(self):
        """SileroVADAnalyzer used in the example must be importable."""
        from pipecat.audio.vad.silero import SileroVADAnalyzer

    def test_stt_import(self):
        """AssemblyAI STT service used in the example must be importable."""
        from pipecat.services.assemblyai.stt import AssemblyAISTTService

    def test_llm_import(self):
        """OpenAI LLM service used in the example must be importable."""
        from pipecat.services.openai.llm import OpenAILLMService


class TestExampleModuleLoads:
    """Verify that the example module loads without errors.

    Missing optional transport deps (daily, fastapi) are mocked so the
    example module can be loaded and inspected in any environment.
    """

    def _load_example(self):
        """Load the example module with missing transport deps mocked."""
        module_mocks = _mock_missing_transport_modules()
        with patch.dict(sys.modules, module_mocks, clear=False):
            spec, module = _import_example_module()
            spec.loader.exec_module(module)
        return module

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_module_loads_successfully(self, mock_client_cls):
        """The example module should load without raising."""
        self._load_example()

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_module_defines_run_bot(self, mock_client_cls):
        """The example must define an async 'run_bot' function."""
        module = self._load_example()
        assert hasattr(module, "run_bot"), "Example must define 'run_bot'"
        assert asyncio.iscoroutinefunction(module.run_bot), "'run_bot' must be async"

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_module_defines_bot_entrypoint(self, mock_client_cls):
        """The example must define an async 'bot' function (Pipecat Cloud entry-point)."""
        module = self._load_example()
        assert hasattr(module, "bot"), "Example must define 'bot'"
        assert asyncio.iscoroutinefunction(module.bot), "'bot' must be async"

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_run_bot_signature(self, mock_client_cls):
        """run_bot must accept (transport, runner_args)."""
        module = self._load_example()
        sig = inspect.signature(module.run_bot)
        params = list(sig.parameters.keys())
        assert "transport" in params, "run_bot must accept a 'transport' parameter"
        assert "runner_args" in params, "run_bot must accept a 'runner_args' parameter"

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_bot_signature(self, mock_client_cls):
        """bot must accept (runner_args)."""
        module = self._load_example()
        sig = inspect.signature(module.bot)
        params = list(sig.parameters.keys())
        assert "runner_args" in params, "bot must accept a 'runner_args' parameter"

    @patch.dict(os.environ, {
        "DEEPDUB_API_KEY": "test-key",
        "DEEPDUB_VOICE_ID": "test-voice",
        "DEEPDUB_MODEL": "dd-etts-2.5",
        "OPENAI_API_KEY": "test-openai-key",
        "ASSEMBLYAI_API_KEY": "test-assemblyai-key",
    })
    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_transport_params_dict(self, mock_client_cls):
        """The example must define transport_params with daily, twilio, webrtc keys."""
        module = self._load_example()
        assert hasattr(module, "transport_params"), "Example must define 'transport_params'"
        tp = module.transport_params
        assert "daily" in tp, "transport_params must include 'daily'"
        assert "twilio" in tp, "transport_params must include 'twilio'"
        assert "webrtc" in tp, "transport_params must include 'webrtc'"

        # Each value should be callable (lambda)
        for key in ("daily", "twilio", "webrtc"):
            assert callable(tp[key]), f"transport_params['{key}'] must be callable"


class TestExampleServiceConstruction:
    """Verify the services are created correctly using the example's pattern."""

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_deepdub_tts_construction_matches_example(self, mock_client_cls):
        """DeepdubTTSService should be constructable the way the example does it."""
        from pipecat_deepdub_tts import DeepdubTTSService

        tts = DeepdubTTSService(
            api_key="test-deepdub-key",
            voice_id="test-voice-id",
            model="dd-etts-2.5",
        )
        assert tts._settings["model"] == "dd-etts-2.5"
        assert tts._settings["voice_prompt_id"] == "test-voice-id"

    def test_openai_llm_construction_matches_example(self):
        """OpenAILLMService should be constructable the way the example does it."""
        from pipecat.services.openai.llm import OpenAILLMService

        llm = OpenAILLMService(api_key="test-openai-key")
        assert llm is not None

    def test_assemblyai_stt_construction_matches_example(self):
        """AssemblyAISTTService should be constructable the way the example does it."""
        from pipecat.services.assemblyai.stt import AssemblyAISTTService

        stt = AssemblyAISTTService(api_key="test-assemblyai-key")
        assert stt is not None


class TestExamplePipelineAssembly:
    """Verify the full pipeline can be assembled as the example does it."""

    @patch("pipecat_deepdub_tts.tts.DeepdubClient")
    def test_pipeline_assembles(self, mock_client_cls):
        """The pipeline should assemble without errors using the example's pattern."""
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        from pipecat.services.openai.llm import OpenAILLMService

        from pipecat_deepdub_tts import DeepdubTTSService

        stt = AssemblyAISTTService(api_key="test-assemblyai-key")

        tts = DeepdubTTSService(
            api_key="test-deepdub-key",
            voice_id="test-voice-id",
            model="dd-etts-2.5",
        )

        llm = OpenAILLMService(api_key="test-openai-key")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ]

        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        # Use a mock transport since we don't have a real one
        mock_transport = MagicMock()
        mock_transport.input.return_value = MagicMock()
        mock_transport.output.return_value = MagicMock()

        pipeline = Pipeline(
            [
                mock_transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                mock_transport.output(),
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

        assert task is not None

    def test_webrtc_transport_params(self):
        """WebRTC TransportParams (always available) should be constructable."""
        from pipecat.transports.base_transport import TransportParams

        params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
        assert isinstance(params, TransportParams)
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True

    @pytest.mark.skipif(not HAVE_DAILY, reason="daily not installed")
    def test_daily_transport_params(self):
        """DailyParams should be constructable when daily is installed."""
        from pipecat.transports.daily.transport import DailyParams

        params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
        assert isinstance(params, DailyParams)
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True

    @pytest.mark.skipif(not HAVE_FASTAPI, reason="fastapi not installed")
    def test_twilio_transport_params(self):
        """FastAPIWebsocketParams should be constructable when fastapi is installed."""
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

        params = FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
        assert isinstance(params, FastAPIWebsocketParams)
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True


class TestExampleLLMContext:
    """Verify the LLM context setup matches what the example does."""

    def test_system_message_is_present(self):
        """The example's system message should be a valid LLM context message."""
        from pipecat.processors.aggregators.llm_context import LLMContext

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate "
                    "your capabilities in a succinct way. Your output will be spoken aloud, "
                    "so avoid special characters that can't easily be spoken, such as emojis "
                    "or bullet points. Respond to what the user said in a creative and helpful way."
                ),
            },
        ]

        context = LLMContext(messages)
        assert context is not None
        assert len(context.messages) == 1
        assert context.messages[0]["role"] == "system"

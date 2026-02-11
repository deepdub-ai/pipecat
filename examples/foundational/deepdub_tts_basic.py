#
# Copyright (c) 2025, Deepdub AI
#
# SPDX-License-Identifier: MIT
#

"""Basic Deepdub TTS example with AssemblyAI STT and OpenAI LLM.

This example demonstrates a full voice conversational AI pipeline using:
- AssemblyAI for speech-to-text (STT)
- OpenAI for the language model (LLM)
- Deepdub AI for text-to-speech (TTS)

Supports multiple transports (Daily, Twilio, WebRTC) via Pipecat's
``create_transport`` utility for easy deployment to Pipecat Cloud.

Required environment variables (see .env.example):
    DEEPDUB_API_KEY     - Deepdub API key
    DEEPDUB_VOICE_ID    - Deepdub voice prompt ID
    DEEPDUB_MODEL       - Deepdub TTS model (e.g. dd-etts-2.5)
    OPENAI_API_KEY      - OpenAI API key
    ASSEMBLYAI_API_KEY  - AssemblyAI API key

Usage:
    pip install pipecat-deepdub-tts "pipecat-ai[assemblyai,openai,silero]"
    python examples/foundational/deepdub_tts_basic.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from pipecat_deepdub_tts import DeepdubTTSService

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = AssemblyAISTTService(api_key=os.getenv("ASSEMBLYAI_API_KEY"))

    tts = DeepdubTTSService(
        api_key=os.getenv("DEEPDUB_API_KEY"),
        voice_id=os.getenv("DEEPDUB_VOICE_ID"),
        model=os.getenv("DEEPDUB_MODEL", "dd-etts-2.5"),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

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
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

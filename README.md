# Pipecat Deepdub TTS

Official [Deepdub AI](https://deepdub.ai/) Text-to-Speech integration for [Pipecat](https://github.com/pipecat-ai/pipecat) -- a framework for building voice and multimodal conversational AI applications.

> **Note**: This integration is maintained by Deepdub AI. As the official provider of the TTS service, we are committed to actively maintaining and updating this integration.

## Pipecat Compatibility

**Tested with Pipecat v0.0.97**

This integration has been tested with Pipecat version 0.0.97. For compatibility with other versions, refer to the [Pipecat changelog](https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md).

## Features

- **Real-time Streaming**: WebSocket-based streaming for low-latency audio generation
- **Multiple Models**: Support for Deepdub TTS models (`dd-etts-2.5`, `dd-etts-3.0`)
- **Voice Customization**: Configurable temperature, variance, tempo, and prompt boost
- **Accent Control**: Blend accents between locales with fine-grained ratio control
- **Flexible Sample Rates**: 8000, 16000, 22050, 24000, 44100, 48000 Hz
- **Interruption Handling**: Clean disconnect/reconnect on user interruption
- **Metrics Support**: Built-in performance tracking and monitoring

## Installation

### Using pip

```bash
pip install pipecat-deepdub-tts
```

### Using uv

```bash
uv add pipecat-deepdub-tts
```

### From source

```bash
git clone https://github.com/deepdub-ai/pipecat-deepdub-tts.git
cd pipecat-deepdub-tts
pip install -e .
```

## Quick Start

### 1. Get Your Deepdub API Key

Sign up at [Deepdub AI](https://deepdub.ai/) and obtain your API key.

### 2. Basic Usage

```python
from pipecat_deepdub_tts import DeepdubTTSService

tts = DeepdubTTSService(
    api_key="your-deepdub-api-key",
    voice_id="your-voice-prompt-id",
    model="dd-etts-2.5",
)
```

### 3. Complete Example with Pipeline

```python
import asyncio
import os
from dotenv import load_dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat_deepdub_tts import DeepdubTTSService

load_dotenv()

async def main():
    tts = DeepdubTTSService(
        api_key=os.getenv("DEEPDUB_API_KEY"),
        voice_id=os.getenv("DEEPDUB_VOICE_ID"),
        model=os.getenv("DEEPDUB_MODEL", "dd-etts-2.5"),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        llm,
        tts,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### DeepdubTTSService Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | Deepdub API key for authentication |
| `voice_id` | `str` | required | Voice prompt ID for TTS synthesis |
| `model` | `str` | `"dd-etts-2.5"` | TTS model name |
| `sample_rate` | `int` | `16000` | Audio sample rate in Hz |
| `params` | `InputParams` | `None` | Optional voice customization parameters |

### InputParams

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `locale` | `str` | `"en-US"` | Language locale for synthesis |
| `temperature` | `float` | `None` | Controls output variability |
| `variance` | `float` | `None` | Controls variance in generated speech |
| `tempo` | `float` | `None` | Speech tempo multiplier |
| `prompt_boost` | `bool` | `None` | Enable prompt boosting for improved quality |
| `accent_base_locale` | `str` | `None` | Base locale for accent control |
| `accent_locale` | `str` | `None` | Target accent locale |
| `accent_ratio` | `float` | `None` | Accent blending ratio (0.0 to 1.0) |

### Example with Custom Configuration

```python
tts = DeepdubTTSService(
    api_key="your-api-key",
    voice_id="your-voice-prompt-id",
    model="dd-etts-3.0",
    sample_rate=24000,
    params=DeepdubTTSService.InputParams(
        locale="en-US",
        temperature=0.7,
        tempo=1.1,
        prompt_boost=True,
        accent_base_locale="en-US",
        accent_locale="fr-FR",
        accent_ratio=0.3,
    ),
)
```

## Environment Variables

Create a `.env` file in your project root:

```env
# Deepdub TTS
DEEPDUB_API_KEY=your_deepdub_api_key_here
DEEPDUB_VOICE_ID=your_voice_prompt_id_here
DEEPDUB_MODEL=dd-etts-2.5

# OpenAI (for LLM in examples)
OPENAI_API_KEY=your_openai_api_key_here

# AssemblyAI (for STT in examples)
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

## Examples

See the [examples](./examples) directory for complete working examples:

- **[deepdub_tts_basic.py](./examples/foundational/deepdub_tts_basic.py)** -- Full pipeline with AssemblyAI STT, OpenAI LLM, and Deepdub TTS

To run the example:

```bash
# Install dependencies
pip install pipecat-deepdub-tts "pipecat-ai[assemblyai,openai,silero]" python-dotenv

# Set up your .env file with API keys (see .env.example)
# Then run
python examples/foundational/deepdub_tts_basic.py
```

## Testing

### Unit tests (no API key needed)

```bash
pytest tests/test_tts.py -k "not Integration"
```

### Integration tests (saves audio to disk)

Requires `DEEPDUB_API_KEY` and `DEEPDUB_VOICE_ID` environment variables:

```bash
# Load .env and run integration tests
pytest tests/test_tts.py -k "Integration" -s
```

Generated audio files are saved to `tests/output/` for manual listening verification.

## Requirements

- Python >= 3.10, < 3.13
- deepdub >= 0.1.20
- pipecat-ai >= 0.0.97, < 0.1.0
- websockets >= 15.0.1, < 16.0
- loguru >= 0.7.3

## Architecture

`DeepdubTTSService` extends Pipecat's `InterruptibleTTSService` base class for WebSocket-based TTS without word-level timestamp alignment. It uses the `DeepdubClient` from the [`deepdub`](https://github.com/deepdub-ai/deepdub) package for WebSocket protocol communication.

The service:
1. Opens a WebSocket to Deepdub's streaming endpoint on pipeline start
2. Sends text chunks as they arrive from the LLM via `async_stream_text()`
3. Receives raw PCM audio (s16le format) via a background task
4. Pushes `TTSAudioRawFrame` frames into the pipeline
5. Handles interruptions by disconnecting and reconnecting

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Support

- GitHub Issues: [pipecat-deepdub-tts/issues](https://github.com/deepdub-ai/pipecat-deepdub-tts/issues)
- Deepdub: [deepdub.ai](https://deepdub.ai/)
- Pipecat Discord: [discord.gg/pipecat](https://discord.gg/pipecat)

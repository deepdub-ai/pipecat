# Pipecat Deepdub

[Deepdub](https://deepdub.ai/) text-to-speech and speech-to-text for [Pipecat](https://github.com/pipecat-ai/pipecat), the open-source framework for voice and multimodal AI agents.

- **`DeepdubTTSService`** streams LLM tokens to Deepdub's emotional TTS (`dd-etts-4.0`) and streams the audio back over one WebSocket. Interruptions cancel the reply in place, without reconnecting.
- **`DeepdubSTTService`** streams microphone audio to Deepdub's real-time speech-to-text and returns interim and final transcripts in English and Hebrew.

This integration is built and maintained by [Deepdub](https://deepdub.ai/).

**Tested with Pipecat v1.17.0.**

## Installation

```bash
uv add pipecat-deepdub
```

or

```bash
pip install pipecat-deepdub
```

Get an API key from [Deepdub](https://deepdub.ai/). Both services take it as `api_key`.

## Usage

```python
import os

from pipecat_deepdub import DeepdubSTTService, DeepdubTTSService

stt = DeepdubSTTService(api_key=os.environ["DEEPDUB_API_KEY"])

tts = DeepdubTTSService(
    api_key=os.environ["DEEPDUB_API_KEY"],
    settings=DeepdubTTSService.Settings(
        voice="b2abb241-ac92-48bf-a890-fec03f43e209",
        language="en-US",
    ),
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
```

Change the voice, locale or model mid-conversation with a `TTSUpdateSettingsFrame`. The TTS service reconfigures the open connection, so the change takes effect from the next text without a reconnect:

```python
from pipecat.frames.frames import TTSUpdateSettingsFrame

await worker.queue_frame(
    TTSUpdateSettingsFrame(delta=DeepdubTTSService.Settings(language="he-IL"))
)
```

### Text-to-speech

`DeepdubTTSService(api_key, url=..., sample_rate=None, enable_logging=None, turn_timeout=10.0, settings=None)`

| Parameter | Default | Description |
|---|---|---|
| `api_key` | required | Deepdub API key. |
| `url` | `wss://wss.deepdub.ai/ws` | Streaming endpoint. Use `wss://wss.eu.deepdub.ai/ws` to process data in the EU. |
| `sample_rate` | pipeline rate | Output rate in Hz. Deepdub streams 8000, 16000, 22050, 24000, 32000, 36000, 44100 or 48000 Hz; other rates are resampled from the next one up. |
| `enable_logging` | Deepdub default (on) | Set to `False` to keep this session's text out of Deepdub's server-side logs. |
| `turn_timeout` | `10.0` | Seconds to wait for audio after a reply is complete before reporting an error. |

`DeepdubTTSService.Settings`:

| Setting | Default | Description |
|---|---|---|
| `model` | `dd-etts-4.0` | Deepdub TTS model. |
| `voice` | `b2abb241-ac92-48bf-a890-fec03f43e209` | Voice prompt ID. |
| `language` | `en-US` | Locale to speak, such as `he-IL` or `fr-FR`. A Pipecat `Language` works too: `Language.HE` maps to `he-IL`. |
| `target_gender` | none | `"male"` or `"female"`, for language-specific handling such as Hebrew diacritics. |
| `accept_emojis` | Deepdub default (off) | Whether flag and emotion emojis switch locale or emotion instead of being spoken. |

### Speech-to-text

`DeepdubSTTService(api_key, url=..., sample_rate=None, vad_force_turn_endpoint=True, should_interrupt=True, settings=None)`

| Parameter | Default | Description |
|---|---|---|
| `api_key` | required | Deepdub API key. |
| `url` | `wss://wsgw.deepdub.ai/transcribe-websocket` | Speech-to-text endpoint. Use `wss://wsgw.eu.deepdub.ai/transcribe-websocket` for the EU. |
| `sample_rate` | pipeline rate | Input rate in Hz. |
| `vad_force_turn_endpoint` | `True` | When `True`, Pipecat's turn strategies end the user's turn and a VAD stop asks Deepdub to finalize the transcript at once. When `False`, Deepdub's endpoint detection ends the turn. |
| `should_interrupt` | `True` | Whether the user speaking interrupts the bot, when Deepdub detects turns. |

`DeepdubSTTService.Settings`:

| Setting | Default | Description |
|---|---|---|
| `model` | `stt-egf` | Deepdub STT model. |
| `language` | auto | `"en"` or `"he"` (or `Language.EN` / `Language.HE`). Unset, Deepdub detects the language of each utterance. |
| `language_hints_strict` | Deepdub default (off) | Recognize only `language`, instead of preferring it. |
| `context_terms` | none | Words and names to recognize reliably, such as product names. |
| `context_text` | none | Background text about the conversation. |
| `max_endpoint_delay_ms` | Deepdub default | Longest wait after speech stops before Deepdub reports an endpoint, 500 to 3000. |
| `client_reference_id` | none | Your own identifier for the session. |

## Running the example

[`examples/foundational/deepdub-voice-agent.py`](examples/foundational/deepdub-voice-agent.py) is a voice agent that listens and speaks with Deepdub and answers with OpenAI.

```bash
git clone https://github.com/deepdub-ai/pipecat.git pipecat-deepdub
cd pipecat-deepdub
uv sync --group example
cp .env.example .env  # then add DEEPDUB_API_KEY and OPENAI_API_KEY
uv run examples/foundational/deepdub-voice-agent.py
```

Open http://localhost:7860/client, allow the microphone, and talk. Interrupt the bot mid-sentence to hear it stop. To run it over Daily or Twilio instead, pass `--transport daily` or `--transport twilio`.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```

The tests run against a local fake of each Deepdub endpoint, so they need no API key.

## Migrating from `pipecat-deepdub-tts`

`pipecat-deepdub` replaces `pipecat-deepdub-tts`, which supported Pipecat 0.0.x.

- Install `pipecat-deepdub` and import from `pipecat_deepdub`.
- `voice_id`, `model` and `params=InputParams(locale=...)` are now `settings=DeepdubTTSService.Settings(voice=..., model=..., language=...)`.
- `temperature`, `variance`, `tempo`, `prompt_boost` and the accent options are gone: Deepdub's streaming endpoint ignores them.

## License

MIT. See [LICENSE](LICENSE).

## Support

- [GitHub issues](https://github.com/deepdub-ai/pipecat/issues)
- [Deepdub API documentation](https://docs.deepdub.ai)
- [Pipecat Discord](https://discord.gg/pipecat)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-26

### Fixed

- `DeepdubSTTService` no longer treats the empty tokens Deepdub sends during silence as
  speech. With a live microphone they pushed an empty interim transcript right after
  each turn ended, which started a new user turn and interrupted every reply.

## [0.1.0] - 2026-11-26

First release of `pipecat-deepdub`, which replaces `pipecat-deepdub-tts`.

### Added

- `DeepdubTTSService` for Pipecat 1.x: streams LLM tokens to Deepdub over one WebSocket
  and streams the audio back. Interruptions cancel the reply without reconnecting,
  replies queued behind it wait their turn, and voice, locale and model changes apply
  to the open connection.
- `DeepdubSTTService`: real-time English and Hebrew transcription with interim results,
  fast finalization on VAD stop, or Deepdub's own turn detection.
- EU endpoints for both services.
- Foundational example `examples/foundational/deepdub-voice-agent.py`.

### Removed

- The `pipecat-deepdub-tts` package and its Pipecat 0.0.x support.
- `temperature`, `variance`, `tempo`, `prompt_boost` and accent options, which Deepdub's
  streaming endpoint ignores.

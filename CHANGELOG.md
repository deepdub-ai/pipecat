# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-11

### Added
- Initial release of Deepdub TTS integration for Pipecat
- WebSocket-based streaming TTS service implementation using `InterruptibleTTSService`
- Support for Deepdub models (`dd-etts-2.5`, `dd-etts-3.0`)
- Configurable voice parameters (temperature, variance, tempo, prompt boost)
- Accent control support (base locale, accent locale, accent ratio)
- Configurable sample rate (8000, 16000, 22050, 24000, 44100, 48000 Hz)
- Metrics and monitoring support
- Error handling with automatic reconnection
- Interruption handling via disconnect/reconnect
- Foundational example with AssemblyAI STT + OpenAI LLM pipeline
- Unit tests and integration test with audio output verification

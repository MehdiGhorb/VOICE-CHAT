# Interruptible Voice Chat Agent (Yes, AI can now interrupt you!)

Real-time voice chat built on LiveKit that can be configured to interrupt and correct speech in-context (for example, grammatical mistakes or factual mistakes).

## Features

- Live voice pipeline: Deepgram (STT) + OpenAI (LLM) + Cartesia (TTS)
- Emotion-aware assistant behavior
- Optional interruption logic during conversation
- Runtime behavior controlled from LiveKit room metadata

## Quick Start (Docker)

1. Create a `.env.local` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
CARTESIA_API_KEY=your_cartesia_key

LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://livekit:7880
```

2. Start LiveKit + agent:

```bash
docker compose up --build
```

3. Connect your client to LiveKit at `ws://localhost:7880` and join a room.

## Interruption Configuration

Pass JSON metadata when creating/joining a room to control interruption behavior:

```json
{
  "enable_interruption": true,
  "interrupt_on_grammatical_mistakes": true,
  "interrupt_on_factual_mistakes": true,
  "interruption_phrase": "Actually",
  "custom_interrupt_prompt": "Politely interrupt when the user states an incorrect fact.",
  "min_endpointing_delay": 0.3,
  "enable_pause_detection": true
}
```

Useful metadata keys:

- `voice` (Cartesia voice ID)
- `model` (default: `gpt-4o-mini`)
- `enable_filling_words` (adds more natural fillers)
- `enable_interruption`
- `interrupt_on_grammatical_mistakes`
- `interrupt_on_factual_mistakes`
- `interruption_phrase`
- `custom_interrupt_prompt`

## Notes

- Interruption behavior relies on VAD (`silero`) and endpointing settings.
- Keep `.env.local` private and never commit real API keys.
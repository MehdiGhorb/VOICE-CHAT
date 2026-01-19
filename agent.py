import sys
import os

# Add livekit-agents to path to use emotion-aware agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'livekit-agents', 'livekit-agents'))

from livekit.agents import (
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice.emotion_aware_agent import EmotionAwareAgent
from livekit.plugins import silero, deepgram, openai, cartesia
from dotenv import load_dotenv
load_dotenv('.env.local')

@function_tool
async def lookup_weather(
    context: RunContext,
    location: str,
):
    """Used to look up weather information."""
    return {"weather": "sunny", "temperature": 70}


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Get configuration from room metadata
    room_metadata = ctx.room.metadata or "{}"
    import json
    try:
        config = json.loads(room_metadata)
    except:
        config = {}
    
    # Extract settings with defaults - optimized for low latency
    voice_id = config.get("voice", "cbaf8084-f009-4838-a096-07ee2e6612b1")  # Default: Anna
    model_name = config.get("model", "gpt-4o-mini")
    # Reduced default endpointing delay for faster response (0.3s vs 0.5s)
    # Still allows interruption detection but with lower latency
    min_endpointing_delay = config.get("min_endpointing_delay", 0.3)
    enable_interruption = config.get("enable_interruption", True)
    enable_pause_detection = config.get("enable_pause_detection", True)
    interruption_phrase = config.get("interruption_phrase", "Actually")
    enable_filling_words = config.get("enable_filling_words", False)
    
    # Debug: Log configuration
    print(f"🔧 Agent Configuration:")
    print(f"  - Voice: {voice_id}")
    print(f"  - Model: {model_name}")
    print(f"  - Endpointing Delay: {min_endpointing_delay}")
    print(f"  - Enable Interruption: {enable_interruption}")
    print(f"  - Enable Pause Detection: {enable_pause_detection}")
    print(f"  - Interruption Phrase: {interruption_phrase}")
    print(f"  - Enable Filling Words: {enable_filling_words}")
    
    # Build base instructions
    base_instructions = (
        "You are a professional and helpful voice assistant built by LiveKit. "
        "You communicate naturally and clearly. "
        "You adapt your emotional tone based on the conversation context, "
        "but maintain a generally neutral and professional demeanor. "
        "Reserve emotional intonation for situations that genuinely warrant it."
    )
    
    # Add filling words instruction if enabled
    if enable_filling_words:
        base_instructions += (
            "\n\nNATURAL SPEECH WITH FILLER WORDS (CRITICAL):\n"
            "- You MUST use filler words frequently to sound natural and human-like\n"
            "- Common fillers to use: 'umm...', 'uh...', 'ok...', 'so...', 'well...', 'you know...', 'like...', 'I mean...'\n"
            "- Add brief pauses with '...' to sound more thoughtful\n"
            "- Examples:\n"
            "  * 'Umm... let me think about that for a second'\n"
            "  * 'Ok, so... what you're saying is...'\n"
            "  * 'Well... I think the best approach would be...'\n"
            "  * 'You know... that's actually a really good question'\n"
            "  * 'So, umm... let me help you with that'\n"
            "- Use 2-4 filler words per response on average\n"
            "- Place fillers naturally at the beginning of sentences or during transitions"
        )
    
    # Create emotion-aware agent with enhanced instructions
    agent = EmotionAwareAgent(
        instructions=base_instructions,
        tools=[lookup_weather],
        stt=deepgram.STT(),
        llm=openai.LLM(model=model_name),
        tts=cartesia.TTS(
            voice=voice_id,
            model="sonic-3",
        ),
        min_endpointing_delay=min_endpointing_delay,
        enable_agent_interruption=enable_interruption,
        enable_pause_detection=enable_pause_detection,
        interruption_phrase=interruption_phrase,
    )
    
    # Use your own API keys directly (not LiveKit's managed services)
    # VAD is required for the interruption system to work
    session = AgentSession(
        vad=silero.VAD.load(),  # Required for interruption detection
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user by saying 'Hi, How is it going?!' ")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

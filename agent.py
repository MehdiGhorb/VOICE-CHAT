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

    # Create emotion-aware agent with enhanced instructions
    agent = EmotionAwareAgent(
        instructions=(
            "You are a professional and helpful voice assistant built by LiveKit. "
            "You communicate naturally and clearly. "
            "You adapt your emotional tone based on the conversation context, "
            "but maintain a generally neutral and professional demeanor. "
            "Reserve emotional intonation for situations that genuinely warrant it."
        ),
        tools=[lookup_weather],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(
            # Use an emotive voice for best emotion control
            # Leo, Jace, Kyle, Gavin are recommended male voices
            # Maya, Tessa, Dana, Marian are recommended female voices
            voice="cbaf8084-f009-4838-a096-07ee2e6612b1",  # Maya - Emotive female voice
            model="sonic-3",  # sonic-3 has best emotion support
        ),
    )
    
    # Use your own API keys directly (not LiveKit's managed services)
    # VAD is required for the interruption system to work
    session = AgentSession(
        vad=silero.VAD.load(),  # Required for interruption detection
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user naturally and ask about their day")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
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

    agent = Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
        tools=[lookup_weather],
    )
    
    # Use your own API keys directly (not LiveKit's managed services)
    # VAD is required for the interruption system to work
    session = AgentSession( 
        vad=silero.VAD.load(),  # Required for interruption detection
        stt=deepgram.STT(),  # Uses DEEPGRAM_API_KEY from .env.local
        llm=openai.LLM(model="gpt-4o-mini"),  # Uses OPENAI_API_KEY from .env.local
        tts=cartesia.TTS(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),  # Uses CARTESIA_API_KEY from .env.local
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

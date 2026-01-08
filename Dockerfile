FROM python:3.11-slim

WORKDIR /app

# Copy the entire livekit-agents directory (custom version)
COPY livekit-agents/ /app/livekit-agents/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgobject-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install custom livekit-agents packages in editable mode
RUN pip install --no-cache-dir -e /app/livekit-agents/livekit-agents

# Install plugin packages
RUN pip install --no-cache-dir \
    -e /app/livekit-agents/livekit-plugins/livekit-plugins-openai \
    -e /app/livekit-agents/livekit-plugins/livekit-plugins-deepgram \
    -e /app/livekit-agents/livekit-plugins/livekit-plugins-cartesia \
    -e /app/livekit-agents/livekit-plugins/livekit-plugins-silero

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent.py .
COPY .env.local .env.local

# Run the agent using the start command that connects to LiveKit server
CMD ["python", "agent.py", "start"]

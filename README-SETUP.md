# Voice Chat Integration Setup

## Quick Start

### 1. Start Backend (LiveKit + Agent)
```bash
cd voice-chat
./start.sh
```

This will:
- Start LiveKit server on `ws://localhost:7880`
- Start the AI agent that connects to LiveKit
- Both services run in Docker containers

### 2. Start Frontend (Next.js)
```bash
cd AI-CALL-AUTOMATION
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. Use Voice Chat
1. Navigate to the Voice Chat page in the dashboard
2. Click the "Talk" button
3. Allow microphone access when prompted
4. Start speaking with the AI assistant
5. Click "Talk" again to end the session

## Architecture

- **Frontend**: Next.js app deployed on Vercel
- **Backend**: Python LiveKit agent + LiveKit server running locally on your laptop
- **Connection**: WebRTC via LiveKit for real-time audio streaming

## Requirements

- Docker Desktop installed and running
- Node.js 18+ for frontend
- Internet connection for Vercel deployment

## Environment Variables

### voice-chat/.env.local
```
OPENAI_API_KEY=your_key
DEEPGRAM_API_KEY=your_key
CARTESIA_API_KEY=your_key
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://livekit:7880
```

### AI-CALL-AUTOMATION/.env.local
```
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://localhost:7880
```

## Troubleshooting

**Connection fails**: Ensure Docker is running and services are started via `./start.sh`

**No audio**: Check browser microphone permissions

**Agent not responding**: Check Docker logs: `docker-compose logs -f agent`

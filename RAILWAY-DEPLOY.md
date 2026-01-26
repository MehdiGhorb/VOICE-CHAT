# Railway Deployment Guide

## Simple 2-Service Setup

Your voice-chat app will run on Railway with:
1. **LiveKit Server** (handles WebRTC connections)
2. **Agent Worker** (your custom Python agent)

## Step-by-Step Deployment

### 1. Create LiveKit Service

1. Go to Railway dashboard → New Project
2. Click "New Service" → "Empty Service"
3. Name it: `livekit-server`
4. Settings → Build:
   - Set **Root Directory**: `voice-chat`
   - Set **Dockerfile Path**: `Dockerfile.livekit`
5. Settings → Variables:
   - Add: `LIVEKIT_KEYS=devkey: secret`
6. Settings → Networking:
   - Click "Generate Domain" (you'll get something like `livekit-server.railway.app`)
   - Copy this URL - you'll need it!
7. Deploy!

### 2. Create Agent Worker Service

1. In same project, click "New Service" → "Empty Service"
2. Name it: `agent-worker`
3. Settings → Build:
   - Set **Root Directory**: `voice-chat`
   - Set **Dockerfile Path**: `Dockerfile` (the default one)
4. Settings → Variables - Add these:
   ```
   OPENAI_API_KEY=your-key-here
   DEEPGRAM_API_KEY=your-key-here
   CARTESIA_API_KEY=your-key-here
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   LIVEKIT_URL=ws://livekit-server.railway.internal:7880
   ```
   
   **Important**: Use `livekit-server.railway.internal:7880` for internal communication!
   
5. Deploy!

### 3. Update Your Frontend

In your Next.js app, update the LiveKit connection URL:

- **Development**: `ws://localhost:7880`
- **Production**: `wss://your-livekit-domain.railway.app` (from step 1.6)

## That's It! 🎉

Your services will:
- Agent connects to LiveKit via internal Railway network
- Frontend connects to LiveKit via public domain
- All using your custom livekit-agents code (not the official version)

## Testing

1. Check LiveKit logs: Should see "server started"
2. Check Agent logs: Should see "connected to LiveKit server"
3. Test from frontend

## Cost Optimization

Railway charges per minute of usage:
- Free tier: $5 credit/month
- After that: ~$0.000231 per GB-hour

Keep services sleeping when not in use!

## Troubleshooting

**Agent can't connect to LiveKit:**
- Verify `LIVEKIT_URL=ws://livekit-server.railway.internal:7880`
- Check both services are deployed

**Frontend can't connect:**
- Use the **public domain** from LiveKit service
- Use `wss://` (not `ws://`) for production

**UDP ports for WebRTC:**
- Railway automatically handles UDP port forwarding
- No manual configuration needed

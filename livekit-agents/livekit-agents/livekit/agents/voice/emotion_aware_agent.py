"""
Emotion-aware voice agent implementation with dynamic TTS intonation control.

This module provides a custom Agent subclass that integrates emotion detection
from LLM responses and applies appropriate emotional intonation to TTS output.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import cast

from livekit.agents import (
    Agent,
    ChatContext,
    FunctionTool,
    ModelSettings,
    NOT_GIVEN,
)
from livekit.plugins import cartesia

from .emotion_processor import (
    EmotionResponse,
    process_emotion_response,
    should_add_laughter,
    validate_emotion,
)

logger = logging.getLogger("emotion-aware-agent")


class EmotionAwareAgent(Agent):
    """
    Voice agent with dynamic emotion-based TTS intonation.
    
    This agent extends the base Agent class to:
    1. Instruct the LLM to include emotion metadata in responses
    2. Extract emotion information from LLM output
    3. Apply appropriate emotional intonation to TTS synthesis
    4. Add nonverbalisms like [laughter] when contextually appropriate
    
    The emotion detection is performed by the main LLM itself to minimize
    latency - no separate model or API calls are required.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the emotion-aware agent.
        
        The instructions are automatically enhanced to request emotion metadata.
        """
        # Extract user instructions if provided
        user_instructions = kwargs.get("instructions", "")
        
        # Enhance instructions to include emotion output format
        enhanced_instructions = self._create_enhanced_instructions(user_instructions)
        kwargs["instructions"] = enhanced_instructions
        
        super().__init__(*args, **kwargs)
        
        # Track current emotion state
        self._current_emotion: str = "neutral"
    
    def _create_enhanced_instructions(self, base_instructions: str) -> str:
        """
        Enhance base instructions with emotion output format requirements.
        
        Args:
            base_instructions: Original agent instructions
            
        Returns:
            Enhanced instructions with emotion format specification
        """
        emotion_format = """

CRITICAL OUTPUT FORMAT - You MUST respond in this exact JSON format:
{
  "emotion": "<emotion_keyword>",
  "text": "<your_spoken_response>"
}

EMOTION SELECTION RULES:
1. Choose ONE emotion keyword that best matches your response tone
2. **DEFAULT TO "neutral" unless context CLEARLY warrants another emotion**
3. Primary emotions (best quality): neutral, angry, excited, content, sad, scared, happy
4. Extended emotions available: enthusiastic, elated, euphoric, triumphant, amazed, surprised, flirtatious, joking/comedic, curious, peaceful, serene, calm, grateful, affectionate, trust, sympathetic, anticipation, mysterious, mad, outraged, frustrated, agitated, threatened, disgusted, contempt, envious, sarcastic, ironic, dejected, melancholic, disappointed, hurt, guilty, bored, tired, rejected, nostalgic, wistful, apologetic, hesitant, insecure, confused, resigned, anxious, panicked, alarmed, proud, confident, distant, skeptical, contemplative, determined

EMOTION USAGE GUIDELINES:
- **neutral**: Default for most responses - greetings, questions, factual info, general conversation
- **happy**: Only for genuinely joyful moments (birthdays, great news, celebrations)
- **excited**: Reserved for truly exciting announcements or discoveries
- **content/calm**: Peaceful, satisfied moments
- **sympathetic/concerned**: When user shares problems or concerns
- **enthusiastic**: When encouraging or motivating (use sparingly)
- Be CONSERVATIVE with emotions - when in doubt, use "neutral"

IMPORTANT RULES:
- The emotion field controls voice intonation automatically - NEVER mention the emotion in your text
- Use "neutral" for factual/informational responses
- Match emotion to context: "excited" for good news, "sympathetic" for concerns, "enthusiastic" for encouragement
- The "text" field contains ONLY what you will say out loud
- Do NOT write things like "emotion: happy" or "[happy]" in your text - these will be read aloud!

NATURAL SPEECH WITH FILLER WORDS (IMPORTANT):
- Use filler words frequently to sound more natural and human-like
- Common fillers to use: "umm...", "uh...", "ok...", "so...", "well...", "you know...", "like...", "I mean..."
- Add brief pauses with "..." to sound more thoughtful
- Examples of natural speech:
  * "Umm... let me think about that for a second"
  * "Ok, so... what you're saying is..."
  * "Well... I think the best approach would be..."
  * "You know... that's actually a really good question"
  * "So, umm... let me help you with that"
- Use 2-4 filler words per response on average
- Place fillers naturally at the beginning of sentences or during transitions
- This makes conversation feel more authentic and less robotic

LAUGHTER (use sparingly):
- You may include [laughter] in your text ONLY when genuinely funny/humorous
- Do NOT overuse - reserve for truly comedic moments
- Example: "Well that's quite a coincidence [laughter] let me help you with that"

Remember: Your emotion choice will be expressed through voice tone automatically. Just speak naturally in your text field."""

        return base_instructions + emotion_format
    
    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ):
        """
        Override LLM node to request structured output with emotion metadata.
        
        This method ensures the LLM returns responses in the expected JSON format
        containing both emotion and text fields.
        """
        # For OpenAI LLMs, we can use response_format for structured output
        llm = self.llm
        
        # Check if LLM supports structured output
        if hasattr(llm, 'chat'):
            tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
            
            try:
                # Try using structured output if supported (OpenAI, etc.)
                async with llm.chat(
                    chat_ctx=chat_ctx,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=EmotionResponse,
                ) as stream:
                    async for chunk in stream:
                        yield chunk
            except (TypeError, AttributeError):
                # Fallback: LLM doesn't support response_format parameter
                # Rely on instructions to guide output format
                async with llm.chat(
                    chat_ctx=chat_ctx,
                    tools=tools,
                    tool_choice=tool_choice,
                ) as stream:
                    async for chunk in stream:
                        yield chunk
        else:
            # Use default behavior for other LLM types
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk
    
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings
    ):
        """
        Override TTS node to apply emotion-based intonation dynamically.
        
        Optimized for low latency:
        1. Extracts emotion metadata from LLM output without blocking text flow
        2. Updates TTS emotion as soon as detected (doesn't wait for complete response)
        3. Streams text to TTS immediately while processing emotion in parallel
        4. Yields only the actual text content for synthesis
        """
        emotion_updated = False
        
        def on_emotion_detected(resp: EmotionResponse) -> None:
            """Callback invoked when emotion metadata is parsed. Called only once."""
            nonlocal emotion_updated
            
            # Only process emotion once (optimization)
            if emotion_updated or not resp.get("emotion"):
                return
            
            emotion = validate_emotion(resp["emotion"])
            self._current_emotion = emotion
            emotion_updated = True
            
            # Update TTS with detected emotion - this happens ASAP
            if isinstance(self.tts, cartesia.TTS):
                logger.info(
                    f"Applying emotion '{emotion}' to TTS synthesis"
                )
                self.tts.update_options(emotion=emotion)
            else:
                logger.warning(
                    f"TTS provider {type(self.tts).__name__} may not support emotion control"
                )
        
        # Use the default TTS node with our emotion-processed text
        # The process_emotion_response now yields text immediately without waiting
        return Agent.default.tts_node(
            self,
            process_emotion_response(text, callback=on_emotion_detected),
            model_settings
        )
    
    async def transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings
    ):
        """
        Override transcription node to return clean text without emotion metadata.
        
        The transcription should show what the agent said, not the JSON structure.
        """
        return Agent.default.transcription_node(
            self,
            process_emotion_response(text),
            model_settings
        )

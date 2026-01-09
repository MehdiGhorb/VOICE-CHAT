# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Emotion processing utilities for dynamic TTS intonation control.
Provides lightweight emotion detection from LLM responses with minimal latency.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable
from typing import Callable, Optional

from pydantic_core import from_json
from typing_extensions import TypedDict


class EmotionResponse(TypedDict):
    """Structured response format from LLM with emotion metadata."""
    emotion: str
    text: str


# Primary emotions supported by Cartesia (as documented - lowercase format)
# Reference: https://docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags
SUPPORTED_EMOTIONS = {
    "happy", "excited", "enthusiastic", "elated", "euphoric", "triumphant",
    "amazed", "surprised", "flirtatious", "joking/comedic", "curious", "content",
    "peaceful", "serene", "calm", "grateful", "affectionate", "trust", "sympathetic",
    "anticipation", "mysterious", "angry", "mad", "outraged", "frustrated", "agitated",
    "threatened", "disgusted", "contempt", "envious", "sarcastic", "ironic", "sad",
    "dejected", "melancholic", "disappointed", "hurt", "guilty", "bored", "tired",
    "rejected", "nostalgic", "wistful", "apologetic", "hesitant", "insecure", "confused",
    "resigned", "anxious", "panicked", "alarmed", "scared", "neutral", "proud", "confident",
    "distant", "skeptical", "contemplative", "determined"
}

# Map of capitalized versions for backward compatibility
EMOTION_CASE_MAP = {
    emotion.lower(): emotion for emotion in [
        "Happy", "Excited", "Enthusiastic", "Elated", "Euphoric", "Triumphant",
        "Amazed", "Surprised", "Flirtatious", "Joking/Comedic", "Curious", "Content",
        "Peaceful", "Serene", "Calm", "Grateful", "Affectionate", "Trust", "Sympathetic",
        "Anticipation", "Mysterious", "Angry", "Mad", "Outraged", "Frustrated", "Agitated",
        "Threatened", "Disgusted", "Contempt", "Envious", "Sarcastic", "Ironic", "Sad",
        "Dejected", "Melancholic", "Disappointed", "Hurt", "Guilty", "Bored", "Tired",
        "Rejected", "Nostalgic", "Wistful", "Apologetic", "Hesitant", "Insecure", "Confused",
        "Resigned", "Anxious", "Panicked", "Alarmed", "Scared", "Neutral", "Proud", "Confident",
        "Distant", "Skeptical", "Contemplative", "Determined"
    ]
}

# Pattern to detect [laughter] nonverbalism
LAUGHTER_PATTERN = re.compile(r'\[laughter\]', re.IGNORECASE)


async def process_emotion_response(
    text: AsyncIterable[str],
    callback: Optional[Callable[[EmotionResponse], None]] = None,
) -> AsyncIterable[str]:
    """
    Process streaming LLM output that includes emotion metadata.
    
    Extracts emotion information from structured JSON response and yields
    only the actual text content for TTS synthesis.
    
    Optimized for low latency:
    - Starts yielding text as soon as possible
    - Caches parsed responses to avoid redundant parsing
    - Only fires callback once when emotion is detected
    
    Args:
        text: Async iterable of text chunks from LLM
        callback: Optional callback invoked when emotion metadata is detected
        
    Yields:
        Text chunks without emotion metadata
    """
    last_response = ""
    acc_text = ""
    emotion_detected = False  # Track if we've already called callback
    last_parsed_resp = None  # Cache last successful parse
    
    async for chunk in text:
        acc_text += chunk
        
        # Try to parse, but use cached result if parsing fails
        try:
            # Try parsing as JSON with partial support (streaming)
            resp: EmotionResponse = from_json(
                acc_text, 
                allow_partial="trailing-strings"
            )
            last_parsed_resp = resp  # Cache successful parse
            
            # Only call callback once when emotion is first detected
            if callback and not emotion_detected and resp.get("emotion"):
                callback(resp)
                emotion_detected = True
        except ValueError:
            # Not yet valid JSON, use cached response if available
            if last_parsed_resp is None:
                continue
            resp = last_parsed_resp
        
        if not resp.get("text"):
            continue
        
        # Yield only new text delta - this enables streaming to TTS immediately
        new_delta = resp["text"][len(last_response):]
        if new_delta:
            yield new_delta
        last_response = resp["text"]


def validate_emotion(emotion: str) -> str:
    """
    Validate and normalize emotion string.
    
    Args:
        emotion: Raw emotion string from LLM (case-insensitive)
        
    Returns:
        Properly cased emotion string or "Neutral" if invalid
    """
    normalized = emotion.lower().strip()
    if normalized in SUPPORTED_EMOTIONS:
        # Return the proper case version for Cartesia API
        return EMOTION_CASE_MAP.get(normalized, "Neutral")
    return "Neutral"


def contains_laughter(text: str) -> bool:
    """
    Check if text contains [laughter] nonverbalism marker.
    
    Args:
        text: Text to check
        
    Returns:
        True if [laughter] is present
    """
    return bool(LAUGHTER_PATTERN.search(text))


def should_add_laughter(text: str, emotion: str) -> bool:
    """
    Determine if [laughter] should be added based on context.
    
    Conservative approach: only add laughter for specific emotions
    and when the response content is appropriate.
    
    Args:
        text: Response text
        emotion: Detected emotion
        
    Returns:
        True if [laughter] should be added
    """
    # Don't add if already present
    if contains_laughter(text):
        return False
    
    # Only add for genuinely humorous/happy emotions
    laugh_emotions = {
        "joking/comedic", "happy", "elated", "euphoric", 
        "excited", "enthusiastic", "flirtatious"
    }
    
    if emotion.lower() not in laugh_emotions:
        return False
    
    # Check for humor indicators in text
    humor_indicators = [
        "haha", "hehe", "lol", "ha ha", "he he",
        "joke", "funny", "hilarious", "amusing"
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in humor_indicators)

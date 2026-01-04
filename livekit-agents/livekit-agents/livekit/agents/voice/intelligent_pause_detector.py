"""
Intelligent Pause Detection for Natural AI Interruptions

This module provides context-aware pause detection that determines:
1. Whether a user has completed a thought (vs mid-sentence pause)
2. Whether the AI has something valuable to contribute right now

No hard-coded timing thresholds - uses speech context and LLM reasoning.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..log import logger

if TYPE_CHECKING:
    from .. import llm


@dataclass
class PauseAnalysis:
    """Result of analyzing a pause in user speech."""
    
    is_complete_thought: bool
    """Whether the user finished a complete idea/sentence."""
    
    confidence: float
    """Confidence score (0.0-1.0) in the completion assessment."""
    
    reason: str
    """Explanation of why this pause was classified this way."""


@dataclass
class InterruptionDecision:
    """Decision on whether the AI should interrupt."""
    
    should_interrupt: bool
    """Whether the AI has something valuable to add right now."""
    
    priority: float
    """Priority/urgency score (0.0-1.0) for interrupting."""
    
    reason: str
    """Explanation of why we should/shouldn't interrupt."""


class IntelligentPauseDetector:
    """
    Detects meaningful pauses using speech context and LLM reasoning.
    
    Avoids hard-coded timing by analyzing:
    - Speech patterns and content
    - Conversation context
    - Intonation cues (via STT confidence/patterns)
    - Whether the pause indicates completion vs hesitation
    """
    
    def __init__(self, llm_model: llm.LLM) -> None:
        self._llm = llm_model
        self._recent_utterances: list[str] = []
        self._conversation_context: list[str] = []
        self._pause_analysis_cache: dict[str, PauseAnalysis] = {}
        
    async def analyze_pause(
        self,
        *,
        current_utterance: str,
        partial_transcript: str,
        recent_transcripts: list[str],
    ) -> PauseAnalysis:
        """
        Analyze if a pause indicates a complete thought using LLM context understanding.
        
        100% context-aware with ZERO hard-coded patterns - relies entirely on LLM.
        
        Args:
            current_utterance: The text spoken before the pause
            partial_transcript: Any in-progress/interim text
            recent_transcripts: Last few complete utterances for context
        """
        # Cache key for avoiding redundant analysis
        cache_key = f"{current_utterance}:{partial_transcript}"
        if cache_key in self._pause_analysis_cache:
            return self._pause_analysis_cache[cache_key]
        
        # Only skip completely empty utterances
        if not current_utterance.strip():
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=1.0,
                reason="Empty utterance - just noise/silence"
            )
        
        # Use LLM for ALL analysis - no hard-coded patterns
        try:
            analysis = await self._llm_analyze_completion(
                current_utterance=current_utterance,
                partial_transcript=partial_transcript,
                recent_transcripts=recent_transcripts,
            )
            
            # Cache the result
            self._pause_analysis_cache[cache_key] = analysis
            
            # Limit cache size
            if len(self._pause_analysis_cache) > 50:
                keys = list(self._pause_analysis_cache.keys())
                for key in keys[:25]:
                    del self._pause_analysis_cache[key]
            
            return analysis
            
        except Exception as e:
            logger.warning(f"[PAUSE-DETECT] LLM analysis failed: {e}")
            # Conservative fallback - assume incomplete to avoid premature interruption
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=0.3,
                reason="Analysis failed - defaulting to incomplete to avoid premature interrupt"
            )
    
    async def _llm_analyze_completion(
        self,
        *,
        current_utterance: str,
        partial_transcript: str,
        recent_transcripts: list[str],
    ) -> PauseAnalysis:
        """
        Use LLM to analyze if the utterance represents a complete thought.
        
        Completely context-aware - understands nuances like:
        - "I was wondering..." (mid-thought, continuing)
        - "I was wondering about that." (complete)
        - Natural speech patterns and intentionality
        """
        from .. import llm as llm_module
        
        # Build rich conversational context
        context = "\n".join(recent_transcripts[-5:]) if recent_transcripts else ""
        
        prompt = (
            f"You are analyzing real-time voice conversation to detect thought completion.\n\n"
            f"RECENT CONVERSATION:\n{context}\n\n"
            f"CURRENT UTTERANCE: \"{current_utterance}\"\n"
            f"PARTIAL/INTERIM TEXT: \"{partial_transcript}\"\n\n"
            f"TASK: Determine if the speaker has FINISHED their current thought/idea or is mid-sentence.\n\n"
            f"KEY CONSIDERATIONS:\n"
            f"1. CONTEXT & INTENT: Is this utterance complete in the flow of conversation?\n"
            f"   - \"I was wondering...\" [PAUSE] = INCOMPLETE (clearly continuing)\n"
            f"   - \"I was wondering about your offer.\" = COMPLETE\n"
            f"   - \"So the thing is...\" [PAUSE] = INCOMPLETE\n"
            f"   - \"The thing is, I agree.\" = COMPLETE\n\n"
            f"2. SPEECH PATTERNS: Natural pauses vs intentional stops\n"
            f"   - Hesitation patterns (um, uh, like) suggest continuation\n"
            f"   - Trailing conjunctions (and, but, so) suggest more coming\n"
            f"   - Complete grammatical structure suggests finished\n\n"
            f"3. SEMANTIC WHOLENESS: Did they express a complete idea?\n"
            f"   - Partial questions/statements = INCOMPLETE\n"
            f"   - Answered questions/complete statements = COMPLETE\n\n"
            f"4. CONVERSATIONAL CONTEXT: What makes sense in this dialogue?\n"
            f"   - Building on previous point = may be incomplete\n"
            f"   - Responding to question = likely complete\n\n"
            f"Respond in this EXACT format:\n"
            f"COMPLETE: [yes/no]\n"
            f"CONFIDENCE: [0.0-1.0]\n"
            f"REASON: [specific explanation based on content and context]"
        )
        
        chat_ctx = llm_module.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "You are an expert in natural speech analysis with deep understanding of "
                "conversational context, intent, and human communication patterns. "
                "You analyze MEANING and CONTEXT, not just grammar patterns."
            )
        )
        chat_ctx.add_message(role="user", content=prompt)
        
        response_text = ""
        async for chunk in self._llm.chat(chat_ctx=chat_ctx):
            if chunk.delta and chunk.delta.content:
                response_text += chunk.delta.content
        
        # Parse response
        return self._parse_completion_response(response_text)
    
    def _parse_completion_response(self, response: str) -> PauseAnalysis:
        """Parse LLM response into PauseAnalysis."""
        lines = response.strip().split("\n")
        
        is_complete = False
        confidence = 0.5
        reason = "Unable to parse response"
        
        for line in lines:
            line = line.strip()
            if line.startswith("COMPLETE:"):
                is_complete = "yes" in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        
        return PauseAnalysis(
            is_complete_thought=is_complete,
            confidence=confidence,
            reason=reason
        )
    
    async def should_agent_interrupt(
        self,
        *,
        pause_analysis: PauseAnalysis,
        conversation_context: list[str],
        agent_instructions: str,
    ) -> InterruptionDecision:
        """
        Determine if the AI has something valuable to add right now.
        
        Only interrupts if:
        - User completed their thought
        - AI has relevant, valuable input
        - Not during personal/emotional explanations
        
        Args:
            pause_analysis: Result from analyze_pause
            conversation_context: Recent conversation for context
            agent_instructions: Agent's current instructions/role
        """
        # If user hasn't finished their thought, don't interrupt
        if not pause_analysis.is_complete_thought:
            return InterruptionDecision(
                should_interrupt=False,
                priority=0.0,
                reason=f"User mid-thought: {pause_analysis.reason}"
            )
        
        # Use LLM to assess if we have something valuable to add
        try:
            decision = await self._llm_assess_value(
                pause_analysis=pause_analysis,
                conversation_context=conversation_context,
                agent_instructions=agent_instructions,
            )
            return decision
            
        except Exception as e:
            logger.warning(f"[PAUSE-DETECT] Value assessment failed: {e}")
            # Conservative fallback - don't interrupt unless confident
            if pause_analysis.confidence >= 0.8:
                return InterruptionDecision(
                    should_interrupt=True,
                    priority=0.5,
                    reason="High confidence completion, conservative interrupt"
                )
            return InterruptionDecision(
                should_interrupt=False,
                priority=0.0,
                reason="Low confidence, avoiding interruption"
            )
    
    async def _llm_assess_value(
        self,
        *,
        pause_analysis: PauseAnalysis,
        conversation_context: list[str],
        agent_instructions: str,
    ) -> InterruptionDecision:
        """
        Use LLM to determine if the agent has something valuable to add.
        
        Completely context-aware decision making - understands:
        - Conversational flow and natural turn-taking
        - Emotional/personal moments where listening is better
        - Questions vs statements vs thinking out loud
        - Whether interrupting would be helpful or disruptive
        """
        from .. import llm as llm_module
        
        context_text = "\n".join(conversation_context[-7:])  # More context for better decisions
        
        prompt = (
            f"You are analyzing a real-time voice conversation to decide if the AI should speak now.\n\n"
            f"AGENT ROLE & CAPABILITIES:\n{agent_instructions}\n\n"
            f"CONVERSATION HISTORY:\n{context_text}\n\n"
            f"PAUSE ANALYSIS: {pause_analysis.reason} (confidence: {pause_analysis.confidence:.2f})\n\n"
            f"DECISION TASK: Should the agent interrupt to speak RIGHT NOW?\n\n"
            f"CRITICAL EVALUATION CRITERIA:\n\n"
            f"1. CONVERSATIONAL CONTEXT & INTENT:\n"
            f"   - Is the user expecting a response right now?\n"
            f"   - Did they ask a direct question?\n"
            f"   - Are they making a statement that requires acknowledgment?\n"
            f"   - Or are they just thinking out loud/processing?\n\n"
            f"2. EMOTIONAL & PERSONAL CONTENT:\n"
            f"   - Is the user sharing something personal or emotional?\n"
            f"   - Are they in the middle of explaining feelings/experiences?\n"
            f"   - Would interrupting feel disrespectful or break their flow?\n\n"
            f"3. NATURAL TURN-TAKING:\n"
            f"   - Is this a natural conversational pause/turn boundary?\n"
            f"   - Or might the user have more to add in a moment?\n"
            f"   - Does the conversation flow suggest it's the agent's turn?\n\n"
            f"4. VALUE ASSESSMENT:\n"
            f"   - Does the agent have something VALUABLE/HELPFUL to add?\n"
            f"   - Would it improve the conversation to speak now?\n"
            f"   - Or is silence/waiting more appropriate?\n\n"
            f"5. CONTEXT AWARENESS:\n"
            f"   - Consider the full conversation arc\n"
            f"   - Understand user's communication style\n"
            f"   - Respect the natural rhythm of this specific dialogue\n\n"
            f"WHEN TO INTERRUPT (speak now):\n"
            f"✓ Direct questions requiring answers\n"
            f"✓ User explicitly seeking input/help\n"
            f"✓ Natural turn-taking boundary with something to contribute\n"
            f"✓ Clarification needed to prevent misunderstanding\n\n"
            f"WHEN NOT TO INTERRUPT (stay silent):\n"
            f"✗ Personal stories or emotional sharing in progress\n"
            f"✗ User thinking out loud or processing\n"
            f"✗ Mid-explanation or building to a point\n"
            f"✗ Might have more to say after brief pause\n"
            f"✗ Silence would be more respectful/valuable\n\n"
            f"Respond in this EXACT format:\n"
            f"INTERRUPT: [yes/no]\n"
            f"PRIORITY: [0.0-1.0 where 1.0 = urgent/important, 0.0 = unnecessary]\n"
            f"REASON: [detailed context-based explanation of your decision]"
        )
        
        chat_ctx = llm_module.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "You are a world-class expert in conversation dynamics, human communication, "
                "and social intelligence. You understand nuance, context, emotional undertones, "
                "and the delicate art of knowing when to speak and when to listen. "
                "You make decisions based on DEEP CONTEXTUAL UNDERSTANDING, not rules or patterns."
            )
        )
        chat_ctx.add_message(role="user", content=prompt)
        
        response_text = ""
        async for chunk in self._llm.chat(chat_ctx=chat_ctx):
            if chunk.delta and chunk.delta.content:
                response_text += chunk.delta.content
        
        return self._parse_interruption_response(response_text)
    
    def _parse_interruption_response(self, response: str) -> InterruptionDecision:
        """Parse LLM response into InterruptionDecision."""
        lines = response.strip().split("\n")
        
        should_interrupt = False
        priority = 0.0
        reason = "Unable to parse response"
        
        for line in lines:
            line = line.strip()
            if line.startswith("INTERRUPT:"):
                should_interrupt = "yes" in line.lower()
            elif line.startswith("PRIORITY:"):
                try:
                    priority = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        
        return InterruptionDecision(
            should_interrupt=should_interrupt,
            priority=priority,
            reason=reason
        )

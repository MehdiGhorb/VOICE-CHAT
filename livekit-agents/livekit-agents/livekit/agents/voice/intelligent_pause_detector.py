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
        Analyze if a pause indicates a complete thought.
        
        Uses speech patterns, context, and linguistic cues - NO timing thresholds.
        
        Args:
            current_utterance: The text spoken before the pause
            partial_transcript: Any in-progress/interim text
            recent_transcripts: Last few complete utterances for context
        """
        # Cache key for avoiding redundant analysis
        cache_key = f"{current_utterance}:{partial_transcript}"
        if cache_key in self._pause_analysis_cache:
            return self._pause_analysis_cache[cache_key]
        
        # Quick heuristic checks (linguistic patterns) - HUMAN-LIKE
        if not current_utterance.strip():
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=1.0,
                reason="Empty utterance - just noise/silence"
            )
        
        utterance_lower = current_utterance.lower().strip()
        words = utterance_lower.split()
        
        # Too short - definitely incomplete
        if len(words) < 3:
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=0.95,
                reason="Too short to be complete"
            )
        
        # Check for obvious filler patterns at the END
        filler_words = ["uh", "um", "er", "ah", "hmm", "like", "you know", "so", "well"]
        last_word = words[-1] if words else ""
        last_two = " ".join(words[-2:]) if len(words) >= 2 else ""
        
        if last_word in filler_words or last_two in filler_words:
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=0.95,
                reason=f"Ends with filler '{last_word}' - clearly continuing"
            )
        
        # Check for strong incomplete patterns ANYWHERE in the utterance
        strong_incomplete = [
            "i was wondering",
            "the thing is",
            "what i mean is", 
            "let me think",
            "so like",
            "you know what",
            "i'm trying to",
            "i want to",
            "i need to",
            "i'd like to",
            "i was thinking",
            "i was going to",
        ]
        
        for pattern in strong_incomplete:
            if pattern in utterance_lower:
                return PauseAnalysis(
                    is_complete_thought=False,
                    confidence=0.90,
                    reason=f"Contains incomplete pattern: '{pattern}'"
                )
        
        # Check if ends with obvious incomplete grammar
        incomplete_endings = [
            "and", "or", "but", "so", "if", "when", "where", "how", "why",
            "that", "which", "who", "because", "since", "while", "although",
            "to", "for", "with", "about", "like", "as"
        ]
        
        if last_word in incomplete_endings:
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=0.85,
                reason=f"Ends with conjunction/preposition '{last_word}'"
            )
        
        # Check for question words without question mark - likely continuing
        question_starts = ["what", "where", "when", "why", "how", "who", "which"]
        first_word = words[0] if words else ""
        if first_word in question_starts and not current_utterance.endswith("?"):
            # Could be incomplete question
            if len(words) < 5:
                return PauseAnalysis(
                    is_complete_thought=False,
                    confidence=0.80,
                    reason=f"Short question starting with '{first_word}' without ending"
                )
        
        # If we have sentence-ending punctuation, likely complete
        if current_utterance.endswith((".", "!", "?")):
            return PauseAnalysis(
                is_complete_thought=True,
                confidence=0.85,
                reason="Ends with sentence-ending punctuation"
            )
        
        # For longer utterances without obvious incompleteness, do quick LLM check
        # But only if it's worth it (mid-length, unclear)
        if len(words) >= 5 and len(words) <= 20:
            try:
                analysis = await self._quick_llm_check(current_utterance, recent_transcripts)
                self._pause_analysis_cache[cache_key] = analysis
                
                # Limit cache size
                if len(self._pause_analysis_cache) > 50:
                    keys = list(self._pause_analysis_cache.keys())
                    for key in keys[:25]:
                        del self._pause_analysis_cache[key]
                
                return analysis
            except Exception:
                # Fallback if LLM fails
                pass
        
        # Default: if no clear signals, assume complete (safer for conversation)
        return PauseAnalysis(
            is_complete_thought=True,
            confidence=0.60,
            reason="No clear incomplete signals - allowing response"
        )
    
    async def _quick_llm_check(
        self,
        current_utterance: str,
        recent_transcripts: list[str],
    ) -> PauseAnalysis:
        """Quick LLM check for ambiguous cases."""
        from .. import llm as llm_module
        
        context = " | ".join(recent_transcripts[-2:]) if recent_transcripts else ""
        
        prompt = (
            f"Is this a complete thought or mid-sentence?\n\n"
            f"Context: {context}\n"
            f"Utterance: \"{current_utterance}\"\n\n"
            f"Answer ONLY: COMPLETE or INCOMPLETE"
        )
        
        chat_ctx = llm_module.ChatContext()
        chat_ctx.add_message(role="system", content="You analyze speech completion. Reply only COMPLETE or INCOMPLETE.")
        chat_ctx.add_message(role="user", content=prompt)
        
        response_text = ""
        async for chunk in self._llm.chat(chat_ctx=chat_ctx):
            if chunk.delta and chunk.delta.content:
                response_text += chunk.delta.content
        
        is_complete = "complete" in response_text.lower() and "incomplete" not in response_text.lower()
        
        return PauseAnalysis(
            is_complete_thought=is_complete,
            confidence=0.75,
            reason=f"LLM quick check: {response_text.strip()}"
        )
    
    async def _llm_analyze_completion(
        self,
        *,
        current_utterance: str,
        partial_transcript: str,
        recent_transcripts: list[str],
    ) -> PauseAnalysis:
        """Use LLM to analyze if the utterance represents a complete thought."""
        from .. import llm as llm_module
        
        # Build context
        context = "\n".join(recent_transcripts[-3:]) if recent_transcripts else ""
        
        prompt = (
            f"Analyze if this spoken utterance represents a COMPLETE thought or if the speaker is mid-sentence:\n\n"
            f"Recent context: {context}\n\n"
            f"Current utterance: \"{current_utterance}\"\n"
            f"Partial/interim: \"{partial_transcript}\"\n\n"
            f"TASK: Determine if the speaker finished their thought or is pausing mid-idea.\n"
            f"Consider:\n"
            f"1. Sentence structure and grammatical completeness\n"
            f"2. Semantic completeness (did they finish the point?)\n"
            f"3. Whether it's a natural pause or hesitation (um, uh, I mean, etc.)\n"
            f"4. Context from previous utterances\n\n"
            f"Respond in this EXACT format:\n"
            f"COMPLETE: [yes/no]\n"
            f"CONFIDENCE: [0.0-1.0]\n"
            f"REASON: [brief explanation]"
        )
        
        chat_ctx = llm_module.ChatContext()
        chat_ctx.add_message(
            role="system",
            content="You are an expert in speech analysis and natural language understanding. Analyze utterances to detect thought completion."
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
    
    def _basic_completion_check(self, utterance: str) -> PauseAnalysis:
        """Fallback basic completion check without LLM."""
        utterance = utterance.strip()
        
        # Check sentence-ending punctuation
        if utterance and utterance[-1] in ".!?":
            return PauseAnalysis(
                is_complete_thought=True,
                confidence=0.7,
                reason="Ends with sentence-ending punctuation"
            )
        
        # Check word count (short utterances unlikely to be complete)
        words = utterance.split()
        if len(words) < 3:
            return PauseAnalysis(
                is_complete_thought=False,
                confidence=0.6,
                reason="Too short to be complete thought"
            )
        
        # Default to incomplete (safer for natural conversation)
        return PauseAnalysis(
            is_complete_thought=False,
            confidence=0.5,
            reason="Unclear - defaulting to incomplete"
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
        """Use LLM to determine if the agent has something valuable to add."""
        from .. import llm as llm_module
        
        context_text = "\n".join(conversation_context[-5:])
        
        prompt = (
            f"You are deciding if an AI assistant should speak right now.\n\n"
            f"AGENT ROLE: {agent_instructions}\n\n"
            f"CONVERSATION:\n{context_text}\n\n"
            f"SITUATION: User just finished speaking ({pause_analysis.reason})\n\n"
            f"QUESTION: Does the agent have something VALUABLE to add RIGHT NOW?\n\n"
            f"Consider:\n"
            f"1. Does the user need/expect a response?\n"
            f"2. Is this a natural turn-taking moment?\n"
            f"3. Would speaking now add value vs letting user continue?\n"
            f"4. Is user sharing personal/emotional content (don't interrupt)?\n"
            f"5. Is user asking a question or making a statement that needs response?\n\n"
            f"DO NOT INTERRUPT if:\n"
            f"- User is sharing personal stories/emotions\n"
            f"- User is thinking out loud\n"
            f"- User might have more to say\n\n"
            f"Respond in EXACT format:\n"
            f"INTERRUPT: [yes/no]\n"
            f"PRIORITY: [0.0-1.0]\n"
            f"REASON: [brief explanation]"
        )
        
        chat_ctx = llm_module.ChatContext()
        chat_ctx.add_message(
            role="system",
            content="You are an expert in conversation dynamics and know when to speak and when to listen."
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

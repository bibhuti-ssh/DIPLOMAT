"""
Phase 2: Positive Counterpart Sampling for DSA-DPO Pairs

This script reads negative sessions from all_sessions.json, and for each session
with error localization:
1. Keeps conversation context up to the error turn
2. Generates 3 new sessions using self-play from that point
3. Applies reinforcement based on error reasoning (what NOT to do)
4. Scores all 3 sessions
5. Picks the max scored session as the positive counterpart
6. Creates negative-positive pairs
7. Extracts key segments that differentiate positive from negative

Output: phase2_output/ directory with all generated sessions, pairs, and segments
"""

import os
import json
import argparse
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Gemini imports for segment extraction
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    try:
        import google.generativeai as genai_legacy
        GEMINI_LEGACY_AVAILABLE = True
    except ImportError:
        GEMINI_LEGACY_AVAILABLE = False

from session_generator import SessionGenerator, DialogueTurn
from session_scorer import (
    SessionScorer,
    load_alignment_matrix,
    score_session_from_dialogue,
)
from llm_judge import LLMJudge
from politeness_scorer import PolitenessScorer
from strategy_classifier import load_trained_gcns_classifier, GcNSClassifier


@dataclass
class PositiveSample:
    """A candidate positive session generated from error turn."""
    session_id: str
    dialogue: List[Dict]
    scores: Dict
    total_score: float
    label: str


@dataclass
class SegmentPair:
    """A segment-level data pair extracted from session pair."""
    segment_id: str
    start_index: int  # Relative to full dialogue (shared_context + continuation)
    end_index: int
    selection_reason: str
    
    # The key differentiating segments
    positive_segment: List[Dict]
    negative_segment: List[Dict]
    
    # Context before the segment (same for both)
    context_before: List[Dict]


@dataclass
class DsaDpoPair:
    """A negative-positive session pair for DSA-DPO training."""
    pair_id: str
    scenario: Dict
    error_turn_index: int
    error_reason: str
    
    # Context before divergence (shared)
    shared_context: List[Dict]
    
    # Negative session (original)
    negative_session_id: str
    negative_continuation: List[Dict]
    negative_scores: Dict
    negative_total_score: float
    
    # Positive session (best of 3)
    positive_session_id: str
    positive_continuation: List[Dict]
    positive_scores: Dict
    positive_total_score: float
    
    # All 3 candidates for reference
    all_positive_candidates: List[Dict]


class Phase2PositiveSampler:
    """
    Generates positive counterpart sessions for negative sessions.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        model_override: Optional[str] = None,
        judge_model_override: Optional[str] = None,
        num_candidates: int = 3,
    ):
        self.config_path = config_path
        self.num_candidates = num_candidates
        self.model_override = model_override
        self.judge_model_override = judge_model_override
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("PHASE 2: Positive Counterpart Sampling + Segment Extraction")
        print("=" * 60)
        
        # Initialize components
        print("\nInitializing components...")
        
        # Load GcNS classifier
        print("Loading GcNS strategy classifier...")
        gcns_model_path = "trained_models/gcns_negotiation_classifier"
        self.gcns_classifier = load_trained_gcns_classifier(model_path=gcns_model_path)
        print(f"✓ GcNS classifier loaded with {len(self.gcns_classifier.strategy_labels)} strategies")
        
        # Initialize generator and scorer
        # model_override controls the employer agent; judge_model_override controls the judge
        self.generator = SessionGenerator(config_path, model_override=model_override)
        self.scorer = SessionScorer(config_path, gcns_classifier=self.gcns_classifier)
        self.judge = LLMJudge(config_path, model_override=judge_model_override or model_override)
        self.politeness_scorer = PolitenessScorer()
        if judge_model_override:
            print(f"✓ Judge model override: {judge_model_override}")
        
        # Initialize segment extraction LLM (Gemini)
        self._init_segment_extractor(model_override)
        
        # Load alignment matrices
        self.alignment_matrices = self._load_alignment_matrices()
        
        # Output directory
        self.output_dir = "dsa_dpo_pipeline/outputs/phase2_output"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "candidates"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "segments"), exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
    
    def _init_segment_extractor(self, model_override: Optional[str] = None):
        """Initialize LLM for segment extraction."""
        self.segment_model = model_override or "gemini-2.5-flash"
        self.segment_client = None
        self.segment_legacy_model = None
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("⚠ GEMINI_API_KEY not set. Segment extraction will fail.")
            return
        
        if GEMINI_AVAILABLE:
            try:
                self.segment_client = genai.Client(api_key=api_key)
                print(f"✓ Segment extractor initialized (New SDK): {self.segment_model}")
            except Exception as e:
                print(f"⚠ Failed to init Gemini new SDK: {e}")
        elif GEMINI_LEGACY_AVAILABLE:
            try:
                genai_legacy.configure(api_key=api_key)
                self.segment_legacy_model = genai_legacy.GenerativeModel(self.segment_model)
                print(f"✓ Segment extractor initialized (Legacy SDK): {self.segment_model}")
            except Exception as e:
                print(f"⚠ Failed to init Gemini legacy SDK: {e}")
    
    def _load_alignment_matrices(self) -> Dict:
        """Load negotiation and persuasion alignment matrices."""
        matrices = {}
        
        neg_path = "dsa_dpo_pipeline/alignment_matrices/negotiation_alignment.csv"
        if os.path.exists(neg_path):
            matrices["negotiation"] = load_alignment_matrix(neg_path)
            print(f"✓ Loaded negotiation alignment matrix: {len(matrices['negotiation'])} pairs")
        
        per_path = "dsa_dpo_pipeline/alignment_matrices/persuasion_alignment.csv"
        if os.path.exists(per_path):
            matrices["persuasion"] = load_alignment_matrix(per_path)
            print(f"✓ Loaded persuasion alignment matrix: {len(matrices['persuasion'])} pairs")
        
        return matrices
    
    def _build_reinforcement_prompt(self, error_reason: str) -> str:
        """
        Build a reinforcement prompt based on error reasoning.
        Tells the employer what NOT to do.
        """
        reinforcement = f"""
IMPORTANT GUIDANCE FOR THIS RESPONSE:
Based on analysis of a suboptimal response in a similar situation, please AVOID the following:
{error_reason}

Instead, focus on:
- Maintaining rapport while being professional
- Acknowledging the other party's perspective before stating your position
- Offering creative alternatives or concessions where possible
- Using collaborative language ("we", "together", "let's explore")
- Being firm on key principles while showing flexibility on execution

Generate a response that achieves the negotiation goal while maintaining a positive relationship.
"""
        return reinforcement.strip()
    
    def extract_key_segment(
        self,
        scenario: Dict,
        negative_dialogue: List[Dict],
        positive_dialogue: List[Dict],
        error_turn_index: int,
    ) -> Optional[SegmentPair]:
        """
        Use LLM to identify the key segment that makes the positive session better.
        
        Args:
            scenario: The negotiation scenario
            negative_dialogue: Full negative session dialogue
            positive_dialogue: Full positive session dialogue
            error_turn_index: Index where sessions diverge
        
        Returns:
            SegmentPair with the extracted segments, or None if extraction fails
        """
        if not self.segment_client and not self.segment_legacy_model:
            print("    ⚠ Segment extractor not available")
            return None
        
        # Build the prompt for segment selection
        prompt = self._build_segment_extraction_prompt(
            scenario=scenario,
            negative_dialogue=negative_dialogue,
            positive_dialogue=positive_dialogue,
            error_turn_index=error_turn_index,
        )
        
        try:
            # Call LLM
            response_text = self._call_segment_llm(prompt)
            
            # Parse response
            segment_info = self._parse_segment_response(response_text)
            
            if not segment_info:
                print("    ⚠ Failed to parse segment response")
                return None
            
            start_idx = segment_info["start_index"]
            end_idx = segment_info["end_index"]
            reason = segment_info["reason"]
            
            # Validate indices
            if start_idx < error_turn_index:
                print(f"    ⚠ Invalid start_index {start_idx} < error_turn_index {error_turn_index}")
                start_idx = error_turn_index
            
            if end_idx >= len(positive_dialogue):
                end_idx = len(positive_dialogue) - 1
            
            if end_idx >= len(negative_dialogue):
                end_idx = len(negative_dialogue) - 1
            
            if start_idx > end_idx:
                print(f"    ⚠ Invalid segment indices: start={start_idx}, end={end_idx}")
                return None
            
            # Extract segments (inclusive range)
            positive_segment = positive_dialogue[start_idx:end_idx + 1]
            negative_segment = negative_dialogue[start_idx:end_idx + 1]
            context_before = positive_dialogue[:start_idx]  # Same for both
            
            segment_pair = SegmentPair(
                segment_id=f"seg_{uuid.uuid4().hex[:8]}",
                start_index=start_idx,
                end_index=end_idx,
                selection_reason=reason,
                positive_segment=positive_segment,
                negative_segment=negative_segment,
                context_before=context_before,
            )
            
            print(f"    ✓ Extracted segment: turns {start_idx}-{end_idx} ({end_idx - start_idx + 1} turns)")
            return segment_pair
            
        except Exception as e:
            print(f"    ⚠ Segment extraction failed: {e}")
            return None
    
    def _build_segment_extraction_prompt(
        self,
        scenario: Dict,
        negative_dialogue: List[Dict],
        positive_dialogue: List[Dict],
        error_turn_index: int,
    ) -> str:
        """Build prompt for LLM to identify key differentiating segment."""
        
        # Format dialogues for the prompt
        def format_dialogue(dialogue: List[Dict]) -> str:
            formatted = []
            for i, turn in enumerate(dialogue):
                role = turn["role"].capitalize()
                content = turn["content"]
                formatted.append(f"[{i}] {role}: {content}")
            return "\n".join(formatted)
        
        negative_str = format_dialogue(negative_dialogue)
        positive_str = format_dialogue(positive_dialogue)
        
        prompt = f"""You are analyzing two HR negotiation conversations to identify the key segment that makes one better than the other.

## Scenario
- Domain: {scenario.get('domain', 'job negotiation')}
- Employer: {scenario.get('employer', 'Employer')}
- Candidate: {scenario.get('candidate', 'Candidate')}
- Employer's Goal: {scenario.get('negotiation_goal', '')}
- Candidate's Position: {scenario.get('current_position', '')}

## Original (Negative) Conversation:
{negative_str}

## Improved (Positive) Conversation:
{positive_str}

## Task
The conversations are IDENTICAL until turn index {error_turn_index}, then they diverge.
The positive conversation achieves better negotiation outcomes (higher goal completion, better relationship).

Select a **closed interval** [start_index, end_index] from the POSITIVE conversation that:
1. Starts at index {error_turn_index} or later (since turns before that are identical)
2. Ends with an EMPLOYER turn (since we're training the employer agent)
3. Contains the KEY turns that make the positive conversation better than the negative one
4. Excludes pleasantries or generic statements that don't directly impact the negotiation outcome
5. Focuses on the strategic difference - what the employer said/did differently that led to better results

The segment should capture the CONTRAST between how the employer handled the same situation differently.

## Output Format
Return a JSON object with exactly these fields:
{{"start_index": <int>, "end_index": <int>, "reason": "<string explaining why this segment is the key differentiator>"}}

Only output the JSON, no other text."""

        return prompt
    
    def _call_segment_llm(self, prompt: str) -> str:
        """Call LLM for segment extraction."""
        if self.segment_client:
            # New SDK
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            )
            response = self.segment_client.models.generate_content(
                model=self.segment_model,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        elif self.segment_legacy_model:
            # Legacy SDK
            generation_config = genai_legacy.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=1024,
            )
            response = self.segment_legacy_model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text.strip()
        else:
            raise RuntimeError("No segment extraction LLM available")
    
    def _parse_segment_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response to extract segment indices."""
        try:
            # Try to extract JSON from response
            # Handle case where LLM might wrap in markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                text = re.sub(r"```json?\s*", "", text)
                text = re.sub(r"```\s*$", "", text)
            
            result = json.loads(text)
            
            # Validate required fields
            if "start_index" not in result or "end_index" not in result:
                return None
            
            return {
                "start_index": int(result["start_index"]),
                "end_index": int(result["end_index"]),
                "reason": result.get("reason", ""),
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    Parse error: {e}")
            print(f"    Response was: {response_text[:200]}...")

            # Fallback: try to extract fields with regex even if JSON is malformed
            try:
                start_match = re.search(r"\"start_index\"\s*:\s*(\d+)", response_text)
                end_match = re.search(r"\"end_index\"\s*:\s*(\d+)", response_text)
                reason_match = re.search(r"\"reason\"\s*:\s*\"([^\"]*)", response_text)

                if not start_match or not end_match:
                    return None

                return {
                    "start_index": int(start_match.group(1)),
                    "end_index": int(end_match.group(1)),
                    "reason": reason_match.group(1).strip() if reason_match else "",
                }
            except Exception:
                return None
    
    def _dict_to_dialogue_turn(self, turn_dict: Dict, turn_idx: int = 0) -> DialogueTurn:
        """Convert a dict turn to DialogueTurn dataclass."""
        return DialogueTurn(
            role=turn_dict["role"],
            content=turn_dict["content"],
            negotiation_strategy=turn_dict.get("negotiation_strategy", ""),
            persuasion_strategy=turn_dict.get("persuasion_strategy", ""),
            turn_index=turn_idx,
        )
    
    def _dialogue_turn_to_dict(self, turn: DialogueTurn) -> Dict:
        """Convert DialogueTurn to dict format."""
        return {
            "role": turn.role,
            "content": turn.content,
            "negotiation_strategy": turn.negotiation_strategy,
            "persuasion_strategy": turn.persuasion_strategy,
        }
    
    def _generate_continuation_from_context(
        self,
        scenario: Dict,
        context: List[Dict],
        error_reason: str,
        candidate_idx: int,
    ) -> Tuple[List[Dict], str]:
        """
        Generate a continuation of the dialogue from the given context.
        
        Args:
            scenario: The negotiation scenario
            context: Dialogue turns up to (but not including) the error turn
            error_reason: Why the original response was suboptimal
            candidate_idx: Index for unique session ID
        
        Returns:
            Tuple of (full_dialogue as list of dicts, session_id)
        """
        session_id = f"pos_{uuid.uuid4().hex[:8]}_{candidate_idx}"
        
        # Build reinforcement prompt
        reinforcement = self._build_reinforcement_prompt(error_reason)
        
        # Convert context dicts to DialogueTurn objects for the generator
        dialogue_turns: List[DialogueTurn] = [
            self._dict_to_dialogue_turn(turn, idx) 
            for idx, turn in enumerate(context)
        ]
        
        # Also keep a dict version for output
        dialogue_dicts = list(context)
        
        # Determine whose turn it is next (should be employer since error was on employer turn)
        if len(context) > 0:
            last_role = context[-1]["role"]
            next_role = "employer" if last_role == "candidate" else "candidate"
        else:
            next_role = "employer"
        
        # Continue the dialogue
        max_turns = self.config["self_play"]["max_turns"]
        remaining_turns = max_turns - len(dialogue_turns)
        
        for turn_idx in range(remaining_turns):
            current_turn_num = len(dialogue_turns)
            
            if next_role == "employer":
                # Generate employer response with reinforcement (only on first turn)
                employer_response = self.generator.generate_employer_response(
                    dialogue=dialogue_turns,
                    scenario=scenario,
                    extra_instructions=reinforcement if turn_idx == 0 else None,
                )
                
                # Create turn objects
                new_turn = DialogueTurn(
                    role="employer",
                    content=employer_response,
                    negotiation_strategy="",
                    persuasion_strategy="",
                    turn_index=current_turn_num,
                )
                dialogue_turns.append(new_turn)
                dialogue_dicts.append(self._dialogue_turn_to_dict(new_turn))
                next_role = "candidate"
            else:
                # Generate candidate response using BC model
                candidate_response = self.generator.generate_candidate_response(
                    dialogue=dialogue_turns,
                    scenario=scenario,
                )

                if candidate_response is None:
                    break  # End of conversation signal

                # Unpack response tuple: (text, strategy_info)
                candidate_text, strategy_info = candidate_response

                # Create turn objects
                new_turn = DialogueTurn(
                    role="candidate",
                    content=candidate_text,
                    negotiation_strategy=strategy_info.get("negotiation_strategy", ""),
                    persuasion_strategy=strategy_info.get("persuasion_strategy", ""),
                    turn_index=current_turn_num,
                )
                dialogue_turns.append(new_turn)
                dialogue_dicts.append(self._dialogue_turn_to_dict(new_turn))
                next_role = "employer"
            
            # Check for natural end
            if self._is_conversation_ended(dialogue_dicts):
                break
        
        return dialogue_dicts, session_id
    
    def _is_conversation_ended(self, dialogue: List[Dict]) -> bool:
        """Check if conversation has reached a natural end."""
        if len(dialogue) < 4:
            return False
        
        last_content = dialogue[-1]["content"].lower()
        end_phrases = [
            "let's finalize", "we have a deal", "agreed", "welcome aboard",
            "accept the offer", "looking forward", "thank you for",
            "pleasure doing business", "we're aligned"
        ]
        return any(phrase in last_content for phrase in end_phrases)
    
    def score_session(self, dialogue: List[Dict], scenario: Dict, session_id: str) -> PositiveSample:
        """
        Score a generated session using the full scoring pipeline.
        """
        # Convert dialogue to required format
        dialogue_dicts = []
        for turn in dialogue:
            dialogue_dicts.append({
                "role": turn["role"],
                "content": turn["content"],
                "negotiation_strategy": turn.get("negotiation_strategy", ""),
                "persuasion_strategy": turn.get("persuasion_strategy", ""),
            })
        
        # Get LLM judge scores
        judge_result = self.judge.evaluate_session(
            dialogue=dialogue_dicts,
            negotiation_goal=scenario.get("negotiation_goal", ""),
            current_position=scenario.get("current_position", ""),
        )
        
        # Add politeness scores
        for turn_dict in dialogue_dicts:
            scored = self.politeness_scorer.score_utterance(
                turn_dict["content"],
                turn_dict.get("persuasion_strategy"),
            )
            turn_dict["politeness_score"] = scored.score
        
        # Compute session score
        session_score = score_session_from_dialogue(
            session_id=session_id,
            dialogue=dialogue_dicts,
            scenario=scenario,
            llm_judge_results={
                "agreement_quality": judge_result.agreement_quality,
                "mutual_satisfaction": judge_result.mutual_satisfaction,
                "conflict_breakdown": judge_result.conflict_breakdown,
            },
            alignment_matrices=self.alignment_matrices,
            gcns_classifier=self.gcns_classifier,
            config_path=self.config_path,
        )
        
        scores = {
            "agreement_quality": judge_result.agreement_quality,
            "mutual_satisfaction": judge_result.mutual_satisfaction,
            "strategy_alignment": session_score.components.strategy_alignment,
            "politeness_trajectory": session_score.components.politeness_trajectory,
            "conflict_breakdown": judge_result.conflict_breakdown,
            "total_score": session_score.total_score,
        }
        
        return PositiveSample(
            session_id=session_id,
            dialogue=dialogue_dicts,
            scores=scores,
            total_score=session_score.total_score,
            label=session_score.label,
        )
    
    def process_negative_session(self, negative_session: Dict) -> Optional[DsaDpoPair]:
        """
        Process a single negative session to generate positive counterparts.
        
        Args:
            negative_session: The negative session with error localization
        
        Returns:
            DsaDpoPair or None if processing fails
        """
        error_turn = negative_session.get("error_turn", {})
        error_index = error_turn.get("index", -1)
        error_reason = error_turn.get("reason", "")
        
        if error_index < 0:
            print(f"  ⚠ No valid error turn for session {negative_session['session_id']}")
            return None
        
        scenario = negative_session["scenario"]
        original_dialogue = negative_session["dialogue"]
        
        # Extract context up to error turn (exclusive)
        shared_context = original_dialogue[:error_index]
        negative_continuation = original_dialogue[error_index:]
        
        print(f"  Context: {len(shared_context)} turns, Error at turn {error_index}")
        
        # Generate N positive candidates
        positive_candidates = []
        
        for i in range(self.num_candidates):
            print(f"    Generating candidate {i+1}/{self.num_candidates}...", end=" ")
            
            try:
                # Generate continuation
                full_dialogue, session_id = self._generate_continuation_from_context(
                    scenario=scenario,
                    context=shared_context,
                    error_reason=error_reason,
                    candidate_idx=i,
                )
                
                # Extract only the continuation part
                continuation = full_dialogue[error_index:]
                
                # Score the full session
                candidate = self.score_session(full_dialogue, scenario, session_id)
                
                print(f"Score: {candidate.total_score:.3f} ({candidate.label})")
                
                positive_candidates.append({
                    "session_id": candidate.session_id,
                    "continuation": continuation,
                    "full_dialogue": candidate.dialogue,
                    "scores": candidate.scores,
                    "total_score": candidate.total_score,
                    "label": candidate.label,
                })
                
            except Exception as e:
                print(f"Failed: {e}")
                continue
        
        if not positive_candidates:
            print(f"  ✗ All candidates failed for session {negative_session['session_id']}")
            return None
        
        # Select best candidate (highest score)
        best_candidate = max(positive_candidates, key=lambda x: x["total_score"])
        
        print(f"  ✓ Best candidate: {best_candidate['session_id']} (score: {best_candidate['total_score']:.3f})")
        
        # Create DSA-DPO pair
        pair = DsaDpoPair(
            pair_id=f"pair_{uuid.uuid4().hex[:8]}",
            scenario=scenario,
            error_turn_index=error_index,
            error_reason=error_reason,
            shared_context=shared_context,
            negative_session_id=negative_session["session_id"],
            negative_continuation=negative_continuation,
            negative_scores=negative_session["scores"],
            negative_total_score=negative_session["scores"]["total_score"],
            positive_session_id=best_candidate["session_id"],
            positive_continuation=best_candidate["continuation"],
            positive_scores=best_candidate["scores"],
            positive_total_score=best_candidate["total_score"],
            all_positive_candidates=positive_candidates,
        )
        
        return pair

    def _process_single_session(self, neg_session: Dict, index: int, total: int) -> Tuple[Optional[DsaDpoPair], Optional[Dict], bool]:
        """
        Process one negative session and optionally extract segment.

        Returns:
            (pair_or_none, segment_record_or_none, success_flag)
        """
        print(f"\n[{index}/{total}] Session: {neg_session['session_id']}")

        try:
            pair = self.process_negative_session(neg_session)
            if not pair:
                return None, None, False

            # Save individual candidate sessions for reference
            candidates_path = os.path.join(
                self.output_dir, "candidates", f"{pair.pair_id}_candidates.json"
            )
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(pair.all_positive_candidates, f, indent=2, ensure_ascii=False)

            # Extract key segment
            print("  Extracting key segment...")
            negative_full = pair.shared_context + pair.negative_continuation
            positive_full = pair.shared_context + pair.positive_continuation

            segment = self.extract_key_segment(
                scenario=pair.scenario,
                negative_dialogue=negative_full,
                positive_dialogue=positive_full,
                error_turn_index=pair.error_turn_index,
            )

            if segment:
                segment_record = {
                    "pair_id": pair.pair_id,
                    "segment": segment,
                }

                # Save individual segment
                seg_path = os.path.join(
                    self.output_dir, "segments", f"{pair.pair_id}_segment.json"
                )
                with open(seg_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "segment_id": segment.segment_id,
                        "start_index": segment.start_index,
                        "end_index": segment.end_index,
                        "selection_reason": segment.selection_reason,
                        "positive_segment": segment.positive_segment,
                        "negative_segment": segment.negative_segment,
                        "context_before": segment.context_before,
                    }, f, indent=2, ensure_ascii=False)
            else:
                segment_record = None

            return pair, segment_record, True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None, None, False

    def _load_existing_pairs(self) -> List[Dict]:
        """Load existing dsa-dpo pairs from disk."""
        pairs_path = os.path.join(self.output_dir, "dsa_dpo_pairs.json")
        if not os.path.exists(pairs_path):
            return []

        try:
            with open(pairs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed to load existing pairs: {e}")
            return []

    def _load_existing_segments(self) -> List[Dict]:
        """Load existing dsa-dpo segment pairs from disk."""
        segments_path = os.path.join(self.output_dir, "dsa_dpo_segment_pairs.json")
        if not os.path.exists(segments_path):
            return []

        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed to load existing segments: {e}")
            return []

    def _backfill_missing_segments(self) -> List[Dict]:
        """
        Backfill missing segments for existing pairs.

        Returns:
            List of segment records in the same shape as run-time segment_pairs.
        """
        existing_pairs = self._load_existing_pairs()
        existing_segments = self._load_existing_segments()

        if not existing_pairs:
            print("ℹ No existing pairs found for backfill")
            return []

        existing_segment_pair_ids = {s.get("pair_id") for s in existing_segments}
        missing_pairs = [p for p in existing_pairs if p.get("pair_id") not in existing_segment_pair_ids]

        if not missing_pairs:
            print("✓ No missing segments to backfill")
            return []

        print(f"🔧 Backfilling segments for {len(missing_pairs)} existing pairs with missing segments")
        backfilled_segments = []

        for i, pair_data in enumerate(missing_pairs, start=1):
            pair_id = pair_data.get("pair_id", "unknown_pair")
            print(f"  [{i}/{len(missing_pairs)}] Backfilling segment for {pair_id}")

            try:
                scenario = pair_data.get("scenario", {})
                error_turn_index = pair_data.get("error_turn_index", 0)
                shared_context = pair_data.get("shared_context", [])
                negative_continuation = pair_data.get("negative", {}).get("continuation", [])
                positive_continuation = pair_data.get("positive", {}).get("continuation", [])

                negative_full = shared_context + negative_continuation
                positive_full = shared_context + positive_continuation

                if not negative_full or not positive_full:
                    print("    ⚠ Missing dialogue content, skipping")
                    continue

                segment = self.extract_key_segment(
                    scenario=scenario,
                    negative_dialogue=negative_full,
                    positive_dialogue=positive_full,
                    error_turn_index=error_turn_index,
                )

                if not segment:
                    print("    ⚠ Segment extraction failed")
                    continue

                backfilled_segments.append({
                    "pair_id": pair_id,
                    "segment": segment,
                })

                # Save individual segment file for traceability
                seg_path = os.path.join(self.output_dir, "segments", f"{pair_id}_segment.json")
                with open(seg_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "segment_id": segment.segment_id,
                        "start_index": segment.start_index,
                        "end_index": segment.end_index,
                        "selection_reason": segment.selection_reason,
                        "positive_segment": segment.positive_segment,
                        "negative_segment": segment.negative_segment,
                        "context_before": segment.context_before,
                    }, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"    ⚠ Backfill failed: {e}")

        print(f"✓ Backfilled {len(backfilled_segments)} missing segments")
        return backfilled_segments
    
    def _load_existing_progress(self) -> set:
        """
        Load already-processed negative session IDs from existing dsa_dpo_pairs.json.
        Returns a set of negative session IDs that have already been processed.
        """
        pairs_path = os.path.join(self.output_dir, "dsa_dpo_pairs.json")
        if not os.path.exists(pairs_path):
            return set()
        
        try:
            with open(pairs_path, "r", encoding="utf-8") as f:
                existing_pairs = json.load(f)
            processed_ids = set(
                p["negative"]["session_id"] for p in existing_pairs
            )
            print(f"✓ Found {len(processed_ids)} already-processed sessions in {pairs_path}")
            return processed_ids
        except Exception as e:
            print(f"⚠ Failed to load existing pairs: {e}")
            return set()

    def run(
        self,
        input_path: str,
        resume: bool = False,
        parallel_sessions: int = 1,
        backfill_missing_segments: bool = True,
    ) -> Dict:
        """
        Run Phase 2 processing on all negative sessions.
        
        Args:
            input_path: Path to all_sessions.json from Phase 1
            resume: If True, skip already-processed sessions and append results
            parallel_sessions: Number of sessions to process in parallel
            backfill_missing_segments: Backfill segments for existing pairs missing segment entries
        
        Returns:
            Summary statistics
        """
        print(f"\nLoading sessions from: {input_path}")
        
        with open(input_path, "r", encoding="utf-8") as f:
            all_sessions = json.load(f)
        
        # Filter to negative sessions with error localization
        negative_sessions = [
            s for s in all_sessions 
            if s.get("label") == "negative" and s.get("error_turn", {}).get("index", -1) >= 0
        ]
        
        print(f"Total sessions: {len(all_sessions)}")
        print(f"Negative sessions with error localization: {len(negative_sessions)}")
        
        # Resume mode: skip already-processed sessions
        already_processed = set()
        if resume:
            already_processed = self._load_existing_progress()
            remaining = [s for s in negative_sessions if s["session_id"] not in already_processed]
            print(f"Already processed: {len(already_processed)}")
            print(f"Remaining to process: {len(remaining)}")
            negative_sessions = remaining
        
        print()
        
        # Process each negative session
        dsa_dpo_pairs = []
        segment_pairs = []
        stats = {
            "total_negative": len(negative_sessions),
            "pairs_created": 0,
            "segments_extracted": 0,
            "segments_backfilled": 0,
            "failures": 0,
        }

        # Before generating new pairs, fill missing segments for already-existing pairs.
        if backfill_missing_segments:
            backfilled_segments = self._backfill_missing_segments()
            segment_pairs.extend(backfilled_segments)
            stats["segments_backfilled"] = len(backfilled_segments)
        
        if parallel_sessions <= 1:
            for i, neg_session in enumerate(tqdm(negative_sessions, desc="Processing negative sessions")):
                pair, segment_record, success = self._process_single_session(
                    neg_session=neg_session,
                    index=i + 1,
                    total=len(negative_sessions),
                )

                if pair:
                    dsa_dpo_pairs.append(pair)
                    stats["pairs_created"] += 1

                if segment_record:
                    segment_pairs.append(segment_record)
                    stats["segments_extracted"] += 1

                if not success:
                    stats["failures"] += 1
        else:
            print(f"🚀 Parallel mode enabled: {parallel_sessions} sessions in flight")
            with ThreadPoolExecutor(max_workers=parallel_sessions) as executor:
                futures = {
                    executor.submit(
                        self._process_single_session,
                        neg_session,
                        i + 1,
                        len(negative_sessions),
                    ): neg_session["session_id"]
                    for i, neg_session in enumerate(negative_sessions)
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing negative sessions"):
                    pair, segment_record, success = future.result()

                    if pair:
                        dsa_dpo_pairs.append(pair)
                        stats["pairs_created"] += 1

                    if segment_record:
                        segment_pairs.append(segment_record)
                        stats["segments_extracted"] += 1

                    if not success:
                        stats["failures"] += 1
        
        # Save all pairs and segments (append if resuming)
        self._save_results(dsa_dpo_pairs, segment_pairs, stats, append=resume)
        
        return stats
    
    def _save_results(self, pairs: List[DsaDpoPair], segment_pairs: List[Dict], stats: Dict, append: bool = False):
        """Save Phase 2 results. If append=True, merge with existing data."""
        
        # Convert new pairs to serializable format
        new_pairs_data = []
        for pair in pairs:
            new_pairs_data.append({
                "pair_id": pair.pair_id,
                "scenario": pair.scenario,
                "error_turn_index": pair.error_turn_index,
                "error_reason": pair.error_reason,
                "shared_context": pair.shared_context,
                "negative": {
                    "session_id": pair.negative_session_id,
                    "continuation": pair.negative_continuation,
                    "scores": pair.negative_scores,
                    "total_score": pair.negative_total_score,
                },
                "positive": {
                    "session_id": pair.positive_session_id,
                    "continuation": pair.positive_continuation,
                    "scores": pair.positive_scores,
                    "total_score": pair.positive_total_score,
                },
                "score_improvement": pair.positive_total_score - pair.negative_total_score,
            })
        
        # If appending, load existing data and merge
        pairs_path = os.path.join(self.output_dir, "dsa_dpo_pairs.json")
        if append and os.path.exists(pairs_path):
            with open(pairs_path, "r", encoding="utf-8") as f:
                existing_pairs = json.load(f)
            print(f"\n📎 Appending {len(new_pairs_data)} new pairs to {len(existing_pairs)} existing pairs")
            pairs_data = existing_pairs + new_pairs_data
        else:
            pairs_data = new_pairs_data
        
        # Save pairs
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(pairs_data, f, indent=2, ensure_ascii=False)
        print(f"✓ DSA-DPO pairs saved: {pairs_path} ({len(pairs_data)} total)")
        
        # Convert new segment pairs
        new_segments_data = []
        for sp in segment_pairs:
            pair_id = sp["pair_id"]
            segment = sp["segment"]
            
            # Find the corresponding pair data (search in new pairs)
            pair_info = next((p for p in pairs_data if p["pair_id"] == pair_id), None)
            
            new_segments_data.append({
                "pair_id": pair_id,
                "scenario": pair_info["scenario"] if pair_info else {},
                "segment_id": segment.segment_id,
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "selection_reason": segment.selection_reason,
                "context_before": segment.context_before,
                "positive_segment": segment.positive_segment,
                "negative_segment": segment.negative_segment,
                "score_improvement": pair_info["score_improvement"] if pair_info else 0,
            })
        
        # If appending, merge with existing segments
        segments_path = os.path.join(self.output_dir, "dsa_dpo_segment_pairs.json")
        if append and os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f:
                existing_segments = json.load(f)
            print(f"📎 Appending {len(new_segments_data)} new segments to {len(existing_segments)} existing")
            segments_data = existing_segments + new_segments_data
        else:
            segments_data = new_segments_data
        
        # Save segment pairs (this is the final DSA-DPO training format)
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, indent=2, ensure_ascii=False)
        print(f"✓ DSA-DPO segment pairs saved: {segments_path} ({len(segments_data)} total)")
        
        # Save summary (reflects total combined data)
        summary = {
            "stats": stats,
            "pairs_count": len(pairs_data),
            "segments_count": len(segments_data),
            "avg_score_improvement": (
                sum(p["score_improvement"] for p in pairs_data) / len(pairs_data)
                if pairs_data else 0
            ),
            "avg_segment_length": (
                sum(s["end_index"] - s["start_index"] + 1 for s in segments_data) / len(segments_data)
                if segments_data else 0
            ),
        }
        summary_path = os.path.join(self.output_dir, "phase2_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PHASE 2 SUMMARY")
        print("=" * 60)
        if append:
            print(f"This run processed: {stats['total_negative']} sessions")
            print(f"New pairs created: {stats['pairs_created']}")
            print(f"New segments extracted: {stats['segments_extracted']}")
            print(f"Segments backfilled from existing pairs: {stats.get('segments_backfilled', 0)}")
            print(f"Failures this run: {stats['failures']}")
            print(f"---")
        print(f"Total DSA-DPO pairs: {len(pairs_data)}")
        print(f"Total segments: {len(segments_data)}")
        if pairs_data:
            print(f"Average score improvement: {summary['avg_score_improvement']:.3f}")
        if segments_data:
            print(f"Average segment length: {summary['avg_segment_length']:.1f} turns")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Positive Counterpart Sampling")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dsa_dpo_pipeline/outputs/sessions/all_sessions.json",
        help="Path to all_sessions.json from Phase 1"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for employer agent (e.g., 'gemini-2.5-flash')"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override model for LLM judge only (e.g., 'gemini-2.5-flash'). If not set, uses --model or config default."
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of positive candidates to generate per negative session"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress: skip already-processed sessions and append new results"
    )
    parser.add_argument(
        "--parallel-sessions",
        type=int,
        default=1,
        help="Number of negative sessions to process in parallel (use small values like 2-4 to avoid rate limits)"
    )
    parser.add_argument(
        "--no-backfill-missing-segments",
        action="store_true",
        help="Disable backfilling segments for existing pairs that are missing in dsa_dpo_segment_pairs.json"
    )
    
    args = parser.parse_args()
    
    if args.model:
        print(f"🔧 Employer model override: {args.model}")
    if args.judge_model:
        print(f"🔧 Judge model override: {args.judge_model}")
    if args.resume:
        print(f"🔧 Resume mode: will skip already-processed sessions")
    if args.parallel_sessions > 1:
        print(f"🔧 Parallel sessions: {args.parallel_sessions}")
    if args.no_backfill_missing_segments:
        print("🔧 Backfill missing segments: disabled")
    
    sampler = Phase2PositiveSampler(
        config_path=args.config,
        model_override=args.model,
        judge_model_override=args.judge_model,
        num_candidates=args.num_candidates,
    )
    
    sampler.run(
        args.input,
        resume=args.resume,
        parallel_sessions=args.parallel_sessions,
        backfill_missing_segments=not args.no_backfill_missing_segments,
    )


if __name__ == "__main__":
    main()

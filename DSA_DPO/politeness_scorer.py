"""
Politeness Scorer for DSA-DPO Pipeline
Computes per-turn politeness scores for trajectory evaluation.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from convokit import Corpus, Speaker, Utterance
    from convokit import PolitenessStrategies
    CONVOKIT_AVAILABLE = True
except ImportError:
    CONVOKIT_AVAILABLE = False
    print("Warning: ConvoKit not installed. Using fallback politeness estimation.")


# Strategy-based politeness mapping (fallback)
STRATEGY_POLITENESS_MAP = {
    # High politeness
    "Rapport Building": 0.85,
    "Concern Addressing": 0.80,
    "Value Alignment": 0.75,
    "Future Vision Alignment": 0.75,
    "Problem-Solving Focus": 0.70,
    "Active Listening": 0.80,
    
    # Moderate politeness
    "Emotional Appeal": 0.60,
    "Credibility & Confidence": 0.55,
    "Data-Driven Persuasion": 0.55,
    "Self-Interest Appeal": 0.50,
    "Reputation Highlighting": 0.55,
    "Principled Negotiation": 0.60,
    "Collaborative Style": 0.70,
    
    # Lower politeness (competitive)
    "Anchoring": 0.40,
    "Door-in-the-Face": 0.35,
    "Competitive Anchoring": 0.35,
    "Market Signaling": 0.50,
    "BATNA Revelation": 0.45,
    "Ultimatum": 0.20,
    "Threat": 0.15,
    
    # Neutral
    "No Strategy": 0.50,
}


@dataclass
class PolitenessResult:
    """Politeness analysis result for a single utterance."""
    text: str
    score: float  # 0.0 to 1.0
    features: Dict[str, float] = None
    method: str = "unknown"  # "convokit" or "strategy_fallback"


class PolitenessScorer:
    """
    Computes politeness scores for utterances.
    Uses ConvoKit when available, falls back to strategy-based estimation.
    """
    
    def __init__(self, use_convokit: bool = True):
        self.use_convokit = use_convokit and CONVOKIT_AVAILABLE
        self.ps_transformer = None
        
        if self.use_convokit:
            try:
                from convokit import TextParser
                print("[DEBUG] Initializing ConvoKit TextParser (spacy)...", flush=True)
                self.parser = TextParser(verbosity=0)
                self.ps_transformer = PolitenessStrategies()
                print("Initialized ConvoKit PolitenessStrategies", flush=True)
            except Exception as e:
                print(f"Failed to initialize ConvoKit: {e}")
                self.use_convokit = False
    
    def score_utterance_convokit(self, text: str) -> PolitenessResult:
        """Score utterance using ConvoKit PolitenessStrategies."""
        if not text or not isinstance(text, str) or not text.strip():
             return self.score_utterance_fallback(str(text) if text else "")

        if not self.use_convokit or self.ps_transformer is None:
            return self.score_utterance_fallback(text)
        
        try:
            # Create minimal corpus
            speaker = Speaker(id="temp_speaker")
            utterance = Utterance(
                id="temp_utt",
                speaker=speaker,
                text=text,
                conversation_id="temp_conv"
            )
            corpus = Corpus(utterances=[utterance])
            
            # Explicitly parse text first (this fixes the NoneType error if spacy wasn't run)
            corpus = self.parser.transform(corpus)
            corpus = self.ps_transformer.transform(corpus)
            
            # Get features
            utt = corpus.get_utterance("temp_utt")
            raw_features = utt.meta.get("politeness_strategies", {})
            
            # ConvoKit uses format: feature_politeness_==FeatureName==
            # Normalize to simple feature names
            features = {}
            for k, v in raw_features.items():
                # Extract feature name from "feature_politeness_==Name=="
                if k.startswith("feature_politeness_==") and k.endswith("=="):
                    clean_name = k[21:-2]  # Remove prefix and suffix
                    features[clean_name] = v
                else:
                    features[k] = v
            
            # Positive markers (polite behaviors) based on ConvoKit's Stanford Politeness Corpus
            # These increase perceived politeness
            positive_markers = [
                "Gratitude",           # "thank you", "thanks"
                "Please",              # "please" anywhere
                "Please_start",        # starts with "please"
                "Hedges",              # hedging language
                "HASHEDGE",            # has hedge markers
                "Factuality",          # factual statements
                "Deference",           # deferential language
                "Indirect_(greeting)", # greetings like "Hello"
                "Indirect_(btw)",      # indirect markers "by the way"
                "Apologizing",         # apologies
                "HASPOSITIVE",         # positive sentiment
                "1st_person_pl.",      # "we" (collaborative)
                "SUBJUNCTIVE",         # subjunctive mood (softer requests)
            ]
            
            # Negative markers (less polite/more direct behaviors)
            negative_markers = [
                "Direct_question",     # direct questions
                "Direct_start",        # starts with direct command
                "2nd_person",          # "you" (can be confrontational)
                "2nd_person_start",    # starts with "you"
                "1st_person_start",    # starts with "I" (self-focused)
                "HASNEGATIVE",         # negative sentiment
                "INDICATIVE",          # indicative mood (more direct)
            ]
            
            pos_count = sum(1 for m in positive_markers if features.get(m, 0) > 0)
            neg_count = sum(1 for m in negative_markers if features.get(m, 0) > 0)
            
            # Weighted scoring based on Stanford Politeness research
            # Using a baseline-adjustment approach for better differentiation
            weighted_pos = 0.0
            weighted_neg = 0.0
            
            # Strong positive markers (high weight) - clear politeness signals
            strong_positive = {
                "Gratitude": 0.15,      # "thank you" - very polite
                "Please": 0.12,         # "please" - polite request marker
                "Please_start": 0.10,   # starts with please
                "Apologizing": 0.12,    # "sorry" - face-saving
                "Deference": 0.10,      # deferential language
            }
            
            # Moderate positive markers
            moderate_positive = {
                "Hedges": 0.06,           # "maybe", "perhaps" - softening
                "HASHEDGE": 0.05,         # hedge markers present
                "Indirect_(greeting)": 0.08,  # "Hello" - rapport building
                "Indirect_(btw)": 0.04,   # "by the way" - indirect
                "HASPOSITIVE": 0.06,      # positive sentiment
                "1st_person_pl.": 0.07,   # "we" - collaborative framing
                "SUBJUNCTIVE": 0.05,      # "could", "would" - softer
                "Factuality": 0.03,       # factual statements
            }
            
            # Strong negative markers (high penalty) - clear directness/rudeness
            strong_negative = {
                "Direct_start": 0.12,     # command at start
                "2nd_person_start": 0.10, # "You..." at start - confrontational
                "HASNEGATIVE": 0.08,      # negative sentiment
            }
            
            # Moderate negative markers
            moderate_negative = {
                "Direct_question": 0.05,  # direct questions (mild)
                "2nd_person": 0.04,       # "you" usage (context dependent)
                "1st_person_start": 0.03, # "I..." at start (self-focused)
                "INDICATIVE": 0.02,       # direct statements
            }
            
            # Calculate weighted scores
            for marker, weight in strong_positive.items():
                if features.get(marker, 0) > 0:
                    weighted_pos += weight
            for marker, weight in moderate_positive.items():
                if features.get(marker, 0) > 0:
                    weighted_pos += weight
                    
            for marker, weight in strong_negative.items():
                if features.get(marker, 0) > 0:
                    weighted_neg += weight
            for marker, weight in moderate_negative.items():
                if features.get(marker, 0) > 0:
                    weighted_neg += weight
            
            # Start from neutral baseline (0.5) and adjust
            # This gives better differentiation for typical professional dialogue
            score = 0.5 + weighted_pos - weighted_neg
            
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))

            return PolitenessResult(
                text=text,
                score=score,
                features=features,
                method="convokit"
            )
            
        except Exception as e:
            print(f"ConvoKit error: {e}")
            return self.score_utterance_fallback(text)
    
    def score_utterance_fallback(
        self,
        text: str,
        strategy: Optional[str] = None,
    ) -> PolitenessResult:
        """
        Fallback politeness scoring using text heuristics and strategy.
        """
        score = 0.5  # Default neutral
        
        # If strategy provided, use it
        if strategy:
            score = STRATEGY_POLITENESS_MAP.get(strategy, 0.5)
        
        # Text-based adjustments
        text_lower = text.lower()
        
        # Positive indicators
        positive_patterns = [
            r"\bthank\b", r"\bplease\b", r"\bappreciate\b",
            r"\bunderstand\b", r"\bhappy to\b", r"\bglad\b",
            r"\bwould be\b", r"\bcould we\b", r"\bmight\b",
        ]
        
        # Negative indicators
        negative_patterns = [
            r"\bmust\b", r"\bhave to\b", r"\bdemand\b",
            r"\binsist\b", r"\bfinal\b", r"\bnon-negotiable\b",
            r"\btake it or\b", r"\bno way\b",
        ]
        
        pos_matches = sum(1 for p in positive_patterns if re.search(p, text_lower))
        neg_matches = sum(1 for p in negative_patterns if re.search(p, text_lower))
        
        # Adjust score
        adjustment = (pos_matches * 0.05) - (neg_matches * 0.08)
        score = max(0.0, min(1.0, score + adjustment))
        
        return PolitenessResult(
            text=text,
            score=score,
            features={"positive_matches": pos_matches, "negative_matches": neg_matches},
            method="strategy_fallback"
        )
    
    def score_utterance(
        self,
        text: str,
        strategy: Optional[str] = None,
    ) -> PolitenessResult:
        """
        Score a single utterance for politeness.
        
        Args:
            text: Utterance text
            strategy: Optional strategy hint for fallback
        
        Returns:
            PolitenessResult with score in [0, 1]
        """
        if self.use_convokit:
            result = self.score_utterance_convokit(text)
            # If ConvoKit gives neutral, use strategy as tiebreaker
            if strategy and 0.4 < result.score < 0.6:
                strategy_score = STRATEGY_POLITENESS_MAP.get(strategy, 0.5)
                result.score = 0.5 * result.score + 0.5 * strategy_score
            return result
        else:
            return self.score_utterance_fallback(text, strategy)
    
    def score_dialogue(
        self,
        dialogue: List[Dict],
    ) -> List[Dict]:
        """
        Score all turns in a dialogue for politeness.
        
        Args:
            dialogue: List of turns with 'role', 'content'/'response', optional 'strategy'
        
        Returns:
            List of dicts with per-turn politeness scores
        """
        results = []
        
        for turn in dialogue:
            text = turn.get("content", turn.get("response", ""))
            strategy = turn.get("persuasion_strategy", turn.get("negotiation_strategy"))
            role = turn.get("role", "unknown")
            result = self.score_utterance(text, strategy)
            
            results.append({
                "role": role,
                "politeness_score": result.score,
                "method": result.method,
            })
        
        return results
    
    def compute_trajectory_scores(
        self,
        dialogue: List[Dict],
    ) -> List[Dict]:
        """
        Compute politeness scores formatted for trajectory calculation.
        
        Following the methodology: ΔP_t = f_P(y_t) - f_P(c_t)
        where y_t is employer response and c_t is candidate utterance.
        
        We pair each employer turn with the PRECEDING candidate turn,
        as we want to see if employer maintains/improves politeness
        relative to the candidate's input.
        
        Returns:
            List of dicts with candidate_politeness and employer_politeness pairs.
        """
        scored = self.score_dialogue(dialogue)
        trajectory = []
        
        # Find pairs of (candidate, employer) based on role
        # Each employer turn is paired with the preceding candidate turn
        prev_candidate_score = None
        
        for i, turn_score in enumerate(scored):
            role = dialogue[i].get("role", "").lower()
            
            if role == "candidate":
                prev_candidate_score = turn_score.get("politeness_score", 0.5)
            elif role == "employer" and prev_candidate_score is not None:
                employer_score = turn_score.get("politeness_score", 0.5)
                trajectory.append({
                    "candidate_politeness": prev_candidate_score,
                    "employer_politeness": employer_score,
                })
                # Don't reset prev_candidate_score - keep it for potential consecutive employer turns
        
        # If dialogue starts with employer (no preceding candidate), 
        # we skip that turn for trajectory as there's nothing to compare to
        
        return trajectory


def get_politeness_label(score: float) -> str:
    """Convert numeric politeness score to categorical label."""
    if score >= 0.8:
        return "Highly Polite"
    elif score >= 0.6:
        return "Polite"
    elif score >= 0.4:
        return "Neutral"
    elif score >= 0.2:
        return "Impolite"
    return "Highly Impolite"

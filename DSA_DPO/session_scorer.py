"""
Session Scorer for DSA-DPO Pipeline
Computes the utility score: Score = α·AQ + β·MS + γ·PT + δ·SA - ε·CF
"""

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import yaml
from strategy_classifier import GcNSClassifier, StrategyAlignmentScorer


def normalize_strategy_for_alignment(
    strategy: str, alignment_matrix: Dict[Tuple[str, str], float]
) -> str:
    """
    Normalize strategy names from classifier output to match alignment matrix keys.

    The trained GcNS classifier outputs individual strategy names, but the alignment
    matrix uses compound names for some strategies. This function maps classifier
    outputs to the correct alignment matrix keys.

    Args:
        strategy: Strategy name from classifier
        alignment_matrix: The alignment matrix to check for valid keys

    Returns:
        Normalized strategy name that exists in alignment matrix
    """
    # Mapping from classifier output to alignment matrix keys
    STRATEGY_NORMALIZATION = {
        # Compound strategies in alignment matrix
        "Collaborative Style": "Collaborative Style / Win-Win Framing",
        "Win-Win Framing": "Collaborative Style / Win-Win Framing",
        "Principled Negotiation": "Principled Negotiation / Data-Driven Justification",
        "Data-Driven Justification": "Principled Negotiation / Data-Driven Justification",
        # Direct mappings (already correct)
        "Active Listening": "Active Listening",
        "Anchoring": "Anchoring",
        "Door-in-the-Face": "Door-in-the-Face",
        "MESO (Multiple Equivalent Simultaneous Offers)": "MESO (Multiple Equivalent Simultaneous Offers)",
        "Reciprocal Concessions": "Reciprocal Concessions",
        "Credibility Assertion": "Credibility Assertion",
        "No Strategy": "No Strategy",
        # Additional strategies that might appear
        "BATNA Revelation": "BATNA Revelation",
        "Ultimatum": "Ultimatum",
    }

    # First try direct normalization
    if strategy in STRATEGY_NORMALIZATION:
        normalized = STRATEGY_NORMALIZATION[strategy]
        # Verify it exists in the matrix (check both candidate and employer positions)
        matrix_strategies = set()
        for c_strat, e_strat in alignment_matrix.keys():
            matrix_strategies.add(c_strat)
            matrix_strategies.add(e_strat)

        if normalized in matrix_strategies:
            return normalized

    # If strategy already exists in matrix, use it as-is
    matrix_strategies = set()
    for c_strat, e_strat in alignment_matrix.keys():
        matrix_strategies.add(c_strat)
        matrix_strategies.add(e_strat)

    if strategy in matrix_strategies:
        return strategy

    # Fallback: return original strategy (will use default 0.5 in alignment lookup)
    return strategy


@dataclass
class ComponentScores:
    """Individual component scores for a session."""

    agreement_quality: float = 0.0
    mutual_satisfaction: float = 0.0
    strategy_alignment: float = 0.0
    politeness_trajectory: float = 0.0
    conflict_breakdown: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "agreement_quality": self.agreement_quality,
            "mutual_satisfaction": self.mutual_satisfaction,
            "strategy_alignment": self.strategy_alignment,
            "politeness_trajectory": self.politeness_trajectory,
            "conflict_breakdown": self.conflict_breakdown,
        }


@dataclass
class TurnAlignment:
    """Turn-level strategy alignment data."""

    turn_index: int
    candidate_strategy: str
    employer_strategy: str
    alignment_score: float  # 0, 0.5, or 1
    confidence: float
    is_critical: bool
    weighted_score: float = 0.0


@dataclass
class SessionScore:
    """Complete session scoring result."""

    session_id: str
    components: ComponentScores
    total_score: float
    label: str  # "positive" or "negative"
    turn_alignments: List[TurnAlignment] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "scores": self.components.to_dict(),
            "total_score": self.total_score,
            "label": self.label,
        }


class SessionScorer:
    """
    Computes utility scores for negotiation sessions.

    Score = α·AQ + β·MS + γ·PT + δ·SA - ε·CF
    """

    def __init__(
        self,
        config_path: str = "dsa_dpo_pipeline/config.yaml",
        gcns_classifier: Optional[GcNSClassifier] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load weights
        weights = self.config["weights"]
        self.alpha = weights["alpha"]
        self.beta = weights["beta"]
        self.gamma = weights["gamma"]
        self.delta = weights["delta"]
        self.epsilon = weights["epsilon"]

        # Load threshold (single threshold at 0.50)
        thresholds = self.config["thresholds"]
        self.threshold = thresholds.get("threshold", 0.50)

        # Load strategy alignment params
        sa_config = self.config["strategy_alignment"]
        self.temporal_decay_lambda = sa_config["temporal_decay_lambda"]
        self.critical_turn_weight = sa_config["critical_turn_weight"]
        self.catastrophic_threshold = sa_config["catastrophic_failure_threshold"]
        self.catastrophic_confidence_min = sa_config["catastrophic_confidence_min"]
        self.early_turn_ratio = sa_config["early_turn_ratio"]

        # Load critical strategies
        self.critical_negotiation = set(
            self.config["critical_strategies"]["negotiation"]
        )
        self.critical_persuasion = set(self.config["critical_strategies"]["persuasion"])

        # Store GcNS classifier (THE KEY ADDITION!)
        self.gcns_classifier = gcns_classifier

    def compute_total_score(self, components: ComponentScores) -> float:
        """Compute weighted utility score."""
        score = (
            self.alpha * components.agreement_quality
            + self.beta * components.mutual_satisfaction
            + self.gamma * components.politeness_trajectory
            + self.delta * components.strategy_alignment
            - self.epsilon * components.conflict_breakdown
        )
        return score

    def label_session(self, score: float) -> str:
        """Assign label based on score threshold (0.50)."""
        if score >= self.threshold:
            return "positive"
        else:
            return "negative"

    def _infer_strategy_from_probabilities(
        self, strategy_probs: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Infer the most likely strategy from probability distribution.
        Uses argmax to select highest probability strategy.

        Args:
            strategy_probs: Dict mapping strategy names to probabilities

        Returns:
            Tuple of (strategy_name, confidence)
        """
        if not strategy_probs:
            return "No Strategy", 0.0

        # argmax: get strategy with highest probability
        best_strategy = max(strategy_probs.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]

    def classify_dialogue_strategies(
        self,
        dialogue: List[Dict],
    ) -> List[Dict]:
        """
        Use GcNS classifier to classify all turns in dialogue.

        THIS IS THE KEY METHOD that bridges raw dialogue to turn_data!

        Args:
            dialogue: List of dialogue turns with 'speaker' and 'text' fields

        Returns:
            List of turn_data dicts with strategy classifications
        """
        if not self.gcns_classifier:
            raise ValueError(
                "GcNS classifier not initialized! "
                "Pass gcns_classifier to SessionScorer constructor."
            )

        turn_data = []
        context = ""

        # Detect role ordering — dialogue may start with employer or candidate
        first_role = dialogue[0].get("role", dialogue[0].get("speaker", "")).lower()
        start_offset = 0
        if first_role == "employer":
            # Employer opens; shift by 1 so pairs are (candidate, employer)
            start_offset = 1

        for i in range(start_offset, len(dialogue) - 1, 2):  # Process turn pairs
            if i + 1 >= len(dialogue):
                break

            candidate_turn = dialogue[i]
            employer_turn = dialogue[i + 1]

            # Safety: verify roles if available
            c_role = candidate_turn.get("role", candidate_turn.get("speaker", "")).lower()
            e_role = employer_turn.get("role", employer_turn.get("speaker", "")).lower()
            if c_role == "employer" and e_role == "candidate":
                candidate_turn, employer_turn = employer_turn, candidate_turn

            # Get text from either 'content' (from session_generator) or 'text' field
            candidate_text = candidate_turn.get(
                "content", candidate_turn.get("text", "")
            )
            employer_text = employer_turn.get("content", employer_turn.get("text", ""))

            # Classify candidate strategy
            candidate_pred = self.gcns_classifier.predict_strategy(
                candidate_text, context
            )

            # Update context with candidate turn
            context = f"{context} {candidate_text}".strip()

            # Classify employer strategy
            employer_pred = self.gcns_classifier.predict_strategy(
                employer_text, context
            )

            # Update context with employer turn
            context = f"{context} {employer_text}".strip()

            # Store classified turn data
            turn_data.append(
                {
                    "candidate_strategy_probs": candidate_pred.probabilities,
                    "employer_strategy_probs": employer_pred.probabilities,
                    "candidate_strategy": candidate_pred.predicted_strategy,
                    "employer_strategy": employer_pred.predicted_strategy,
                    "candidate_confidence": candidate_pred.confidence,
                    "employer_confidence": employer_pred.confidence,
                }
            )

        return turn_data

    def compute_strategy_alignment(
        self,
        turn_data: List[Dict],
        alignment_matrix: Dict[Tuple[str, str], float],
    ) -> Tuple[float, List[TurnAlignment]]:
        """
        Compute dialogue-level strategy alignment score using RcNS-style approach.

        This follows the document's methodology:
        1. Use GcNS classifier to get strategy probabilities
        2. Use alignment matrix for normative judgment
        3. Weight by confidence (P_c × P_e)
        4. Apply temporal decay and critical turn weighting
        5. Check for catastrophic failures

        Args:
            turn_data: List of turns with strategy classifications
            alignment_matrix: (candidate_strategy, employer_strategy) -> score

        Returns:
            Tuple of (alignment_score, list of TurnAlignment)
        """
        if not turn_data:
            return 0.5, []

        alignments = []
        total_turns = len(turn_data)
        first_turn = 0

        for i, turn in enumerate(turn_data):
            # STEP 1: Get strategies (already classified or from probabilities)
            if "candidate_strategy_probs" in turn and "employer_strategy_probs" in turn:
                # Use classifier probabilities (preferred method)
                c_strat, c_conf = self._infer_strategy_from_probabilities(
                    turn["candidate_strategy_probs"]
                )
                e_strat, e_conf = self._infer_strategy_from_probabilities(
                    turn["employer_strategy_probs"]
                )
            else:
                # Fallback: use pre-determined strategies
                c_strat = turn.get("candidate_strategy", "No Strategy")
                e_strat = turn.get("employer_strategy", "No Strategy")
                c_conf = turn.get("candidate_confidence", 1.0)
                e_conf = turn.get("employer_confidence", 1.0)

            # STEP 2: Look up alignment in matrix (normative judgment)
            # Normalize strategy names to match alignment matrix keys
            c_strat_normalized = normalize_strategy_for_alignment(
                c_strat, alignment_matrix
            )
            e_strat_normalized = normalize_strategy_for_alignment(
                e_strat, alignment_matrix
            )

            key = (c_strat_normalized, e_strat_normalized)
            alignment = alignment_matrix.get(key, 0.5)

            # When either side is "No Strategy", use neutral alignment
            # instead of the harsh 0.0 the matrix assigns
            if c_strat == "No Strategy" or e_strat == "No Strategy":
                alignment = max(alignment, 0.5)

            # STEP 3: Confidence weighting (RcNS-style)
            # Use geometric mean instead of product to avoid overly harsh
            # discounting when probabilities are spread across 11 classes
            confidence = math.sqrt(c_conf * e_conf)

            # STEP 4: Check if critical turn
            is_critical = (
                c_strat in self.critical_negotiation
                or c_strat in self.critical_persuasion
            )

            turn_alignment = TurnAlignment(
                turn_index=i,
                candidate_strategy=c_strat,
                employer_strategy=e_strat,
                alignment_score=alignment,
                confidence=confidence,
                is_critical=is_critical,
            )
            alignments.append(turn_alignment)

        # STEP 5: Weighted aggregation with temporal decay
        weighted_sum = 0.0
        weight_total = 0.0
        catastrophic_failure = False

        for i, ta in enumerate(alignments):
            # Critical turn weight
            w_t = self.critical_turn_weight if ta.is_critical else 1.0

            # Temporal decay (early turns matter more)
            d_t = math.exp(-self.temporal_decay_lambda * (i - first_turn))

            # Turn-level alignment with confidence weighting
            # Alignment_t = Align(Ŝ_c, Ŝ_e) × Confidence(t)
            a_t = ta.alignment_score * ta.confidence
            ta.weighted_score = a_t

            # Accumulate weighted scores
            weighted_sum += w_t * d_t * a_t
            weight_total += w_t * d_t

            # STEP 6: Check for catastrophic failure
            early_threshold = int(total_turns * self.early_turn_ratio)
            if (
                i < early_threshold
                and ta.is_critical
                and ta.alignment_score < 0.5
                and ta.confidence >= self.catastrophic_confidence_min
            ):
                catastrophic_failure = True

        # Compute final dialogue-level alignment score
        if weight_total > 0:
            delta = weighted_sum / weight_total
        else:
            delta = 0.5

        # Apply catastrophic failure override
        if catastrophic_failure:
            delta = min(delta, self.catastrophic_threshold)

        return delta, alignments

    def compute_politeness_trajectory(
        self,
        politeness_scores: List[Dict],
    ) -> float:
        """
        Compute politeness trajectory score.

        Args:
            politeness_scores: List of per-turn politeness
                [{"candidate_politeness": float, "employer_politeness": float}, ...]

        Returns:
            PT score in [0, 1]
        """
        if not politeness_scores:
            return 0.5

        total_turns = len(politeness_scores)
        penalty_sum = 0.0
        weight_sum = 0.0

        for t, scores in enumerate(politeness_scores):
            c_pol = scores.get("candidate_politeness", 0.5)
            e_pol = scores.get("employer_politeness", 0.5)

            # Politeness delta
            delta_p = e_pol - c_pol

            # Temporal weight (cosine: early matters more)
            w_t = math.cos((math.pi / 2) * (t / max(total_turns, 1)))
            w_t = max(w_t, 0.1)  # Minimum weight

            # Penalize only drops
            penalty = w_t * max(0, -delta_p)
            penalty_sum += penalty
            weight_sum += w_t

        # Compute PT score
        if weight_sum > 0:
            pt = 1 - (penalty_sum / weight_sum)
        else:
            pt = 0.5

        return max(0.0, min(1.0, pt))

    def score_session(
        self,
        session_id: str,
        aq_score: float,
        ms_score: float,
        cf_score: float,
        turn_data: List[Dict],
        politeness_scores: List[Dict],
        alignment_matrix: Dict[Tuple[str, str], float],
    ) -> SessionScore:
        """
        Compute complete session score.

        Args:
            session_id: Unique session identifier
            aq_score: Agreement Quality from LLM judge
            ms_score: Mutual Satisfaction from LLM judge
            cf_score: Conflict/Breakdown from LLM judge
            turn_data: Strategy data per turn
            politeness_scores: Politeness data per turn
            alignment_matrix: Strategy alignment lookup

        Returns:
            SessionScore with all components and final label
        """
        # Compute strategy alignment
        sa_score, alignments = self.compute_strategy_alignment(
            turn_data, alignment_matrix
        )

        # Compute politeness trajectory
        pt_score = self.compute_politeness_trajectory(politeness_scores)

        # Build components
        components = ComponentScores(
            agreement_quality=aq_score,
            mutual_satisfaction=ms_score,
            strategy_alignment=sa_score,
            politeness_trajectory=pt_score,
            conflict_breakdown=cf_score,
        )

        # Compute total score
        total = self.compute_total_score(components)

        # Label
        label = self.label_session(total)

        return SessionScore(
            session_id=session_id,
            components=components,
            total_score=total,
            label=label,
            turn_alignments=alignments,
        )


def load_alignment_matrix(csv_path: str) -> Dict[Tuple[str, str], float]:
    """
    Load alignment matrix from CSV.

    Expected format:
    ,Employer_Strategy_1,Employer_Strategy_2,...
    Candidate_Strategy_1,High,Low,...
    Candidate_Strategy_2,Medium,High,...

    Returns:
        Dict mapping (candidate_strategy, employer_strategy) -> score
    """
    import csv

    label_to_score = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
    matrix = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        employer_strategies = header[1:]  # Skip first cell

        for row in reader:
            candidate_strategy = row[0]
            for i, value in enumerate(row[1:]):
                employer_strategy = employer_strategies[i]
                score = label_to_score.get(value.strip(), 0.5)
                matrix[(candidate_strategy, employer_strategy)] = score

    return matrix


def score_session_from_dialogue(
    session_id: str,
    dialogue: List[Dict],
    scenario: Dict,
    llm_judge_results: Dict,
    alignment_matrices: Dict[str, Dict],
    gcns_classifier: GcNSClassifier,
    config_path: str = "dsa_dpo_pipeline/config.yaml",
) -> SessionScore:
    """
    High-level function to score a complete dialogue session.

    THIS IS THE MAIN ENTRY POINT with full GcNS integration!

    Args:
        session_id: Unique identifier
        dialogue: List of dialogue turns [{"speaker": str, "text": str}, ...]
        scenario: Scenario context
        llm_judge_results: Results from LLM-as-judge (AQ, MS, CF)
        alignment_matrices: Both negotiation and persuasion matrices
        gcns_classifier: Trained GcNS strategy classifier
        config_path: Path to config YAML

    Returns:
        SessionScore with complete analysis
    """
    # Initialize scorer with classifier
    scorer = SessionScorer(config_path, gcns_classifier=gcns_classifier)

    # STEP 1: Use GcNS classifier to classify all strategies
    turn_data = scorer.classify_dialogue_strategies(dialogue)

    # STEP 2: Extract politeness scores from dialogue
    # (You need a separate politeness classifier for this)
    politeness_scores = []
    for i in range(0, len(dialogue) - 1, 2):
        if i + 1 < len(dialogue):
            politeness_scores.append(
                {
                    "candidate_politeness": dialogue[i].get("politeness_score", 0.5),
                    "employer_politeness": dialogue[i + 1].get("politeness_score", 0.5),
                }
            )

    # STEP 3: Get alignment matrix
    alignment_matrix = alignment_matrices.get("negotiation", {})

    # STEP 4: Score the session
    return scorer.score_session(
        session_id=session_id,
        aq_score=llm_judge_results.get("agreement_quality", 0.5),
        ms_score=llm_judge_results.get("mutual_satisfaction", 0.5),
        cf_score=llm_judge_results.get("conflict_breakdown", 0.0),
        turn_data=turn_data,
        politeness_scores=politeness_scores,
        alignment_matrix=alignment_matrix,
    )


# test this using :


# Example usage
if __name__ == "__main__":
    from strategy_classifier import load_trained_gcns_classifier

    # Load GcNS classifier
    classifier = load_trained_gcns_classifier(model_path="path/to/trained/roberta-gcns")

    # Load alignment matrix
    alignment_matrix = load_alignment_matrix("alignment_matrix.csv")

    # Example dialogue
    dialogue = [
        {
            "speaker": "candidate",
            "text": "I prefer 90% research focus.",
            "politeness_score": 0.8,
        },
        {
            "speaker": "employer",
            "text": "We can structure onboarding around that.",
            "politeness_score": 0.9,
        },
        {
            "speaker": "candidate",
            "text": "That sounds reasonable.",
            "politeness_score": 0.85,
        },
        {
            "speaker": "employer",
            "text": "Great! Let's finalize details.",
            "politeness_score": 0.9,
        },
    ]

    # LLM judge results
    llm_results = {
        "agreement_quality": 0.8,
        "mutual_satisfaction": 0.75,
        "conflict_breakdown": 0.1,
    }

    # Score session
    result = score_session_from_dialogue(
        session_id="session_001",
        dialogue=dialogue,
        scenario={},
        llm_judge_results=llm_results,
        alignment_matrices={"negotiation": alignment_matrix},
        gcns_classifier=classifier,
    )
    print(f"here is the result {result}\n \n")
    print(f"Session Score: {result.total_score:.3f}")
    print(f"Label: {result.label}")
    print(f"Strategy Alignment: {result.components.strategy_alignment:.3f}")

"""
GcNS: RoBERTa-based Negotiation Strategy Classifier
Used to classify dialogue turns into negotiation strategies with confidence scores.
"""

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import json
import os


@dataclass
class StrategyPrediction:
    """Strategy classification result for a single turn."""

    text: str
    predicted_strategy: str
    confidence: float
    probabilities: Dict[str, float]
    top_k_strategies: List[Tuple[str, float]]


class GcNSClassifier:
    """
    RoBERTa-based Contextual Negotiation Strategy Classifier.

    This classifier identifies negotiation strategies in dialogue turns
    and provides probability distributions over strategy types.
    """

    def __init__(
        self,
        model_path: str,
        strategy_labels: List[str],
        device: Optional[str] = None,
    ):
        """
        Initialize the GcNS classifier.

        Args:
            model_path: Path to fine-tuned RoBERTa model
            strategy_labels: List of strategy names (order must match training)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Strategy labels (must match model training)
        self.strategy_labels = strategy_labels
        self.num_strategies = len(strategy_labels)

        print(f"GcNS Classifier loaded on {self.device}")
        print(f"Strategies: {', '.join(strategy_labels)}")

    def predict_strategy(
        self,
        text: str,
        context: Optional[str] = None,
        return_top_k: int = 3,
    ) -> StrategyPrediction:
        """
        Classify a dialogue turn into negotiation strategies.

        Args:
            text: The utterance to classify
            context: Optional dialogue context (previous turns)
            return_top_k: Number of top strategies to return

        Returns:
            StrategyPrediction with probabilities and top-k strategies
        """
        # Prepare input
        if context:
            input_text = f"{context} </s></s> {text}"
        else:
            input_text = text

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # Create probability dictionary
        prob_dict = {
            strategy: float(prob) for strategy, prob in zip(self.strategy_labels, probs)
        }

        # Get predicted strategy (argmax)
        predicted_idx = np.argmax(probs)
        predicted_strategy = self.strategy_labels[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Get top-k strategies
        top_k_indices = np.argsort(probs)[::-1][:return_top_k]
        top_k_strategies = [
            (self.strategy_labels[idx], float(probs[idx])) for idx in top_k_indices
        ]

        return StrategyPrediction(
            text=text,
            predicted_strategy=predicted_strategy,
            confidence=confidence,
            probabilities=prob_dict,
            top_k_strategies=top_k_strategies,
        )

    def predict_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
        batch_size: int = 16,
    ) -> List[StrategyPrediction]:
        """
        Classify multiple dialogue turns in batches.

        Args:
            texts: List of utterances to classify
            contexts: Optional list of dialogue contexts
            batch_size: Batch size for processing

        Returns:
            List of StrategyPrediction objects
        """
        if contexts is None:
            contexts = [None] * len(texts)

        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_contexts = contexts[i : i + batch_size]

            for text, context in zip(batch_texts, batch_contexts):
                pred = self.predict_strategy(text, context)
                predictions.append(pred)

        return predictions

    def compute_rcns_reward(
        self,
        context_strategy_probs: Dict[str, float],
        response_strategy_probs: Dict[str, float],
        gamma_c: float = 1.0,
    ) -> float:
        """
        Compute RcNS reward for strategy consistency.

        RcNS = G_cNS(C_t) - γ_c * G_cNS(r)

        This measures self-consistency (not opponent adaptation).
        Not directly used for Strategy Alignment, but useful for RL training.

        Args:
            context_strategy_probs: Strategy probabilities from context
            response_strategy_probs: Strategy probabilities from response
            gamma_c: Penalization factor (≥ 1.0)

        Returns:
            RcNS reward score
        """
        # Get max probability from each
        max_context_prob = max(context_strategy_probs.values())
        max_response_prob = max(response_strategy_probs.values())

        # Compute RcNS
        rcns = max_context_prob - gamma_c * max_response_prob

        return rcns


class StrategyAlignmentScorer:
    """
    Computes Strategy Alignment using GcNS classifier + alignment matrix.

    This is what you actually need for DSA-DPO!
    """

    def __init__(
        self,
        classifier: GcNSClassifier,
        alignment_matrix: Dict[Tuple[str, str], float],
    ):
        """
        Initialize scorer with classifier and alignment rules.

        Args:
            classifier: Trained GcNS classifier
            alignment_matrix: (candidate_strategy, employer_strategy) -> score
        """
        self.classifier = classifier
        self.alignment_matrix = alignment_matrix

    def classify_turn_pair(
        self,
        candidate_text: str,
        employer_text: str,
        context: Optional[str] = None,
    ) -> Dict:
        """
        Classify both candidate and employer turns.

        Args:
            candidate_text: Candidate's utterance
            employer_text: Employer's utterance
            context: Optional dialogue context

        Returns:
            Dict with strategies and probabilities for both speakers
        """
        # Classify candidate turn
        candidate_pred = self.classifier.predict_strategy(candidate_text, context)

        # Update context with candidate turn for employer classification
        updated_context = f"{context or ''} {candidate_text}".strip()

        # Classify employer turn
        employer_pred = self.classifier.predict_strategy(employer_text, updated_context)

        return {
            "candidate_strategy": candidate_pred.predicted_strategy,
            "candidate_confidence": candidate_pred.confidence,
            "candidate_strategy_probs": candidate_pred.probabilities,
            "employer_strategy": employer_pred.predicted_strategy,
            "employer_confidence": employer_pred.confidence,
            "employer_strategy_probs": employer_pred.probabilities,
        }

    def score_turn_alignment(
        self,
        candidate_strategy: str,
        employer_strategy: str,
        candidate_confidence: float,
        employer_confidence: float,
    ) -> Tuple[float, float]:
        """
        Score a single turn's strategy alignment.

        Args:
            candidate_strategy: Predicted candidate strategy
            employer_strategy: Predicted employer strategy
            candidate_confidence: Confidence in candidate strategy
            employer_confidence: Confidence in employer strategy

        Returns:
            Tuple of (alignment_score, combined_confidence)
        """
        # Step 1: Look up normative alignment
        key = (candidate_strategy, employer_strategy)
        alignment = self.alignment_matrix.get(key, 0.5)

        # Step 2: Compute confidence weighting
        confidence = candidate_confidence * employer_confidence

        # Step 3: Weighted alignment score
        weighted_alignment = alignment * confidence

        return weighted_alignment, confidence


def load_trained_gcns_classifier(
    model_path: str,
    config_path: Optional[str] = None,
) -> GcNSClassifier:
    """
    Load a trained GcNS classifier from checkpoint.

    Automatically loads strategy labels from strategy_mapping.json in the model directory.

    Args:
        model_path: Path to saved model (directory containing model.safetensors and strategy_mapping.json)
        config_path: Optional path to config with strategy labels (overrides auto-detection)

    Returns:
        Initialized GcNSClassifier
    """
    # First, try to load from strategy_mapping.json in model directory
    mapping_path = os.path.join(model_path, "strategy_mapping.json")

    if os.path.exists(mapping_path):
        print(f"Loading strategy mapping from {mapping_path}")
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            strategies = mapping["id_to_strategy"]
            print(f"Loaded {len(strategies)} strategies: {', '.join(strategies)}")
    elif config_path:
        # Fallback to config if provided
        import yaml

        print(f"Loading strategy labels from config: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            strategies = config.get("strategy_labels", [])
            if not strategies:
                raise ValueError(f"No strategy_labels found in config: {config_path}")
    else:
        # No mapping found - error out
        raise FileNotFoundError(
            f"strategy_mapping.json not found in {model_path}. "
            f"Please ensure your trained model includes this file, or provide config_path."
        )

    return GcNSClassifier(
        model_path=model_path,
        strategy_labels=strategies,
    )


# Example usage
if __name__ == "__main__":
    # Load classifier
    classifier = load_trained_gcns_classifier(
        model_path="path/to/trained/roberta-gcns",
    )

    # Example dialogue turn
    context = "Candidate: I'm looking for a role focused on research."
    candidate_text = (
        "I prefer the role to remain at least 90% focused on pure research."
    )
    employer_text = (
        "Perhaps we can structure onboarding so research remains the priority."
    )

    # Classify candidate strategy
    candidate_pred = classifier.predict_strategy(candidate_text, context)
    print(f"Candidate Strategy: {candidate_pred.predicted_strategy}")
    print(f"Confidence: {candidate_pred.confidence:.3f}")
    print(f"Top-3: {candidate_pred.top_k_strategies}")

    # Classify employer strategy
    updated_context = f"{context} {candidate_text}"
    employer_pred = classifier.predict_strategy(employer_text, updated_context)
    print(f"\nEmployer Strategy: {employer_pred.predicted_strategy}")
    print(f"Confidence: {employer_pred.confidence:.3f}")
    print(f"Top-3: {employer_pred.top_k_strategies}")

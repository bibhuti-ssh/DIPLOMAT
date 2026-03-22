"""
Strategy validation and normalization utilities for annotation scripts.
This ensures LLMs use only the predefined strategy names.
"""

from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches

# Canonical strategy lists - exported for use by annotators
PERSUASION_STRATEGIES = [
    "Rapport Building",
    "Concern Addressing",
    "Emotional Appeal",
    "Credibility & Confidence",
    "Data-Driven Persuasion",
    "Problem-Solving Focus",
    "Self-Interest Appeal",
    "Value Alignment",
    "Reputation Highlighting",
    "Future Vision Alignment",
    "No Strategy",
]

NEGOTIATION_STRATEGIES = [
    "Collaborative Style",
    "Active Listening",
    "Win-Win Framing",
    "Principled Negotiation",
    "Data-Driven Justification",
    "MESO (Multiple Equivalent Simultaneous Offers)",
    "Anchoring",
    "Door-in-the-Face",
    "Reciprocal Concessions",
    "Credibility Assertion",
    "No Strategy",
]

# Strategy definitions for reference
PERSUASION_STRATEGIES_WITH_DEFINITIONS = [
    {"Strategy": "Rapport Building", "Definition": "Establishing trust and a positive relationship to enhance receptiveness."},
    {"Strategy": "Concern Addressing", "Definition": "Actively listening to and resolving objections or hesitations."},
    {"Strategy": "Emotional Appeal", "Definition": "Connecting through emotions such as empathy, excitement, or urgency."},
    {"Strategy": "Credibility & Confidence", "Definition": "Demonstrating expertise and confidence to increase trust and influence."},
    {"Strategy": "Data-Driven Persuasion", "Definition": "Using evidence, facts, and benchmarks to strengthen arguments."},
    {"Strategy": "Problem-Solving Focus", "Definition": "Presenting ideas as solutions that address mutual challenges."},
    {"Strategy": "Self-Interest Appeal", "Definition": "Framing arguments around how the outcome directly benefits the other party."},
    {"Strategy": "Value Alignment", "Definition": "Linking your proposal to the other party's core values and principles."},
    {"Strategy": "Reputation Highlighting", "Definition": "Leveraging past achievements or organizational standing to reinforce trustworthiness."},
    {"Strategy": "Future Vision Alignment", "Definition": "Connecting the proposal with shared long-term goals and aspirations."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
]

NEGOTIATION_STRATEGIES_WITH_DEFINITIONS = [
    {"Strategy": "Collaborative Style", "Definition": "Working jointly to find mutually beneficial outcomes and preserve relationships."},
    {"Strategy": "Active Listening", "Definition": "Demonstrating attentiveness and understanding to build rapport and trust."},
    {"Strategy": "Win-Win Framing", "Definition": "Framing negotiation as a shared problem to solve for mutual benefit."},
    {"Strategy": "Principled Negotiation", "Definition": "Focusing on mutual interests and objective standards rather than positions."},
    {"Strategy": "Data-Driven Justification", "Definition": "Supporting negotiation points with evidence like market benchmarks and past performance."},
    {"Strategy": "MESO (Multiple Equivalent Simultaneous Offers)", "Definition": "Proposing multiple offers of equal value to reveal priorities and increase agreement likelihood."},
    {"Strategy": "Anchoring", "Definition": "Setting a strong initial offer to influence the negotiation range."},
    {"Strategy": "Door-in-the-Face", "Definition": "Starting with a larger request to make the actual target seem more acceptable."},
    {"Strategy": "Reciprocal Concessions", "Definition": "Offering small concessions to encourage reciprocation from the other party."},
    {"Strategy": "Credibility Assertion", "Definition": "Building trust by reinforcing personal or organizational credibility during negotiations."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit negotiation strategy."},
]


def normalize_strategy_name(strategy: str, valid_strategies: List[str]) -> Optional[str]:
    """
    Normalize and validate a strategy name.
    
    Args:
        strategy: Strategy name from LLM output
        valid_strategies: List of valid strategy names
    
    Returns:
        Normalized strategy name if valid, None otherwise
    """
    if not strategy or not strategy.strip():
        return None
    
    # Clean the strategy name
    cleaned = strategy.strip()
    
    # Exact match (case-sensitive)
    if cleaned in valid_strategies:
        return cleaned
    
    # Case-insensitive exact match
    for valid in valid_strategies:
        if cleaned.lower() == valid.lower():
            return valid
    
    # Fuzzy match (handles minor typos)
    matches = get_close_matches(cleaned, valid_strategies, n=1, cutoff=0.85)
    if matches:
        return matches[0]
    
    # No match found
    return None


def validate_and_fix_strategies(labeled_conversation: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Validate and fix strategy names in labeled conversation.
    
    Args:
        labeled_conversation: List of turns with strategy labels
    
    Returns:
        Tuple of (fixed_conversation, invalid_turn_indices)
    """
    invalid_indices = []
    
    for i, turn in enumerate(labeled_conversation):
        neg_strategy = turn.get('negotiation_strategy', '').strip()
        pers_strategy = turn.get('persuasion_strategy', '').strip()
        
        has_invalid = False
        
        # Validate and normalize negotiation strategy
        if neg_strategy:
            normalized_neg = normalize_strategy_name(neg_strategy, NEGOTIATION_STRATEGIES)
            if normalized_neg:
                turn['negotiation_strategy'] = normalized_neg
            else:
                print(f"   ⚠️ Turn {i}: Invalid negotiation strategy '{neg_strategy}'")
                turn['negotiation_strategy'] = ''
                has_invalid = True
        else:
            has_invalid = True
        
        # Validate and normalize persuasion strategy
        if pers_strategy:
            normalized_pers = normalize_strategy_name(pers_strategy, PERSUASION_STRATEGIES)
            if normalized_pers:
                turn['persuasion_strategy'] = normalized_pers
            else:
                print(f"   ⚠️ Turn {i}: Invalid persuasion strategy '{pers_strategy}'")
                turn['persuasion_strategy'] = ''
                has_invalid = True
        else:
            has_invalid = True
        
        if has_invalid:
            invalid_indices.append(i)
    
    return labeled_conversation, invalid_indices


def get_strategy_validation_prompt_addition() -> str:
    """
    Get additional prompt text to emphasize exact strategy name usage.
    """
    return f"""
    CRITICAL - STRATEGY NAME REQUIREMENTS:
    ⚠️ You MUST use ONLY the EXACT strategy names from the provided lists
    ⚠️ PERSUASION strategies come from the PERSUASION list ONLY
    ⚠️ NEGOTIATION strategies come from the NEGOTIATION list ONLY
    ⚠️ Do NOT mix up the two lists
    ⚠️ Do NOT create new strategy names
    ⚠️ Do NOT create new strategy names or variations
    ⚠️ Do NOT use similar but different names (e.g., "Collaboration Style" instead of "Collaborative Style")
    ⚠️ Copy the strategy name EXACTLY as it appears in the list, including capitalization and punctuation
    
    Valid Persuasion Strategies (use EXACTLY as shown):
    {', '.join([f'"{s}"' for s in PERSUASION_STRATEGIES])}
    
    Valid Negotiation Strategies (use EXACTLY as shown):
    {', '.join([f'"{s}"' for s in NEGOTIATION_STRATEGIES])}
    """


def get_persuasion_prompt() -> str:
    """Get prompt section for persuasion strategy annotation."""
    import json
    return f"""# PERSUASION STRATEGIES WITH DEFINITIONS:
{json.dumps(PERSUASION_STRATEGIES_WITH_DEFINITIONS, indent=2)}

Valid strategy names (use EXACTLY):
{json.dumps(PERSUASION_STRATEGIES)}
"""


def get_negotiation_prompt() -> str:
    """Get prompt section for negotiation strategy annotation."""
    import json
    bucket = "\n".join([f"{i+1}. {s}" for i, s in enumerate(NEGOTIATION_STRATEGIES)])
    return f"""# NEGOTIATION STRATEGY BUCKET (choose EXACTLY from this list):
{bucket}

# STRATEGY DEFINITIONS:
{json.dumps(NEGOTIATION_STRATEGIES_WITH_DEFINITIONS, indent=2)}
"""

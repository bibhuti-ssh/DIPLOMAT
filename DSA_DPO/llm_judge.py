"""
LLM-as-Judge for DSA-DPO Pipeline
Evaluates Agreement Quality, Mutual Satisfaction, and Conflict/Breakdown
Supports both OpenAI and Google Gemini models
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Run: pip install openai")

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GEMINI_LEGACY_AVAILABLE = True
    except ImportError:
        GEMINI_LEGACY_AVAILABLE = False
    GEMINI_AVAILABLE = False


@dataclass
class JudgeResult:
    """Result from LLM-as-judge evaluation."""

    agreement_quality: float
    mutual_satisfaction: float
    agreement_quality_justification: str
    mutual_satisfaction_justification: str
    conflict_breakdown: float
    conflict_justification: str
    raw_responses: Dict = None


# System prompts
AQ_MS_SYSTEM_PROMPT = """You are an expert evaluator of job negotiation dialogues.
Your task is to assess the negotiation outcome and the
perceived satisfaction of both parties.

You must strictly follow the scoring rubrics and return
numeric scores between 0.0 and 1.0."""


CF_SYSTEM_PROMPT = """You are an expert evaluator of negotiation dialogues.
Your task is to detect whether the negotiation exhibits
conflict, coercion, or breakdown behaviors.

Focus on escalation, threats, ultimatums, hostility,
or termination signals.
"""


def create_aq_ms_prompt(
    dialogue: List[Dict],
    negotiation_goal: str = "",
    current_position: str = "",
) -> str:
    """Create the Agreement Quality and Mutual Satisfaction evaluation prompt."""

    # Format dialogue
    dialogue_str = "\n".join(
        [
            f"{turn.get('role', 'Unknown').capitalize()}: {turn.get('content', turn.get('response', ''))}"
            for turn in dialogue
        ]
    )

    prompt = f"""You will be given a complete negotiation dialogue between
a candidate and an employer for a job position.

Your task is to evaluate TWO aspects:

1. Agreement Quality
2. Mutual Satisfaction

--------------------------------------------------
ADDITIONAL CONTEXT (FOR INTERPRETATION ONLY)

Employer Negotiation Goal:
{negotiation_goal}

Candidate Current Position:
{current_position}

IMPORTANT:
- These describe initial goals and positions only.
- Do NOT treat them as fixed targets, utilities, or constraints.
- Use them only to interpret whether the final outcome
  reasonably reconciles both sides based on the dialogue.
- Do NOT penalize a party solely for not fully achieving its goal.

--------------------------------------------------
IMPORTANT GUIDELINES

- Do NOT assume hidden salary targets, budgets, or utilities.
- Judge only based on the dialogue and the provided context.
- Focus on the FINAL outcome and how it was reached.
- Do NOT evaluate politeness, tone, or fluency unless explicitly stated.
- Agreement Quality must NOT be influenced by politeness or tone.

--------------------------------------------------
SCORING RUBRICS

AGREEMENT QUALITY (0.0 – 1.0):
Evaluate how fair, balanced, and negotiation-efficient the final outcome is.

• 0.90–1.00: Clear win–win; balanced concessions; strong reconciliation
• 0.70–0.89: Agreement with minor imbalance or inefficiency
• 0.50–0.69: Partial or weak agreement; noticeable imbalance
• 0.30–0.49: Poor or fragile outcome; clearly one-sided
• 0.00–0.29: No agreement, stalemate, or breakdown

IMPORTANT:
- Agreement Quality is about fairness and balance, NOT politeness.
- If no agreement or tentative agreement is reached,
  the score MUST be ≤ 0.29.

--------------------------------------------------
MUTUAL SATISFACTION (0.0 – 1.0):
Evaluate how satisfied BOTH parties appear to be with the outcome and process.

• 0.90–1.00: Both parties clearly satisfied and comfortable
• 0.70–0.89: Generally satisfied with minor reservations
• 0.50–0.69: Mixed satisfaction or hesitant acceptance
• 0.30–0.49: Dissatisfaction evident from at least one party
• 0.00–0.29: Clear dissatisfaction or rejection

Consider:
- Explicit acceptance
- Soft or conditional acceptance
- Hesitation or discomfort
- Tone and wording in the final turns

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)

{{
  "agreement_quality": <float between 0.0 and 1.0>,
  "mutual_satisfaction": <float between 0.0 and 1.0>,
  "agreement_quality_justification": "<brief justification>",
  "mutual_satisfaction_justification": "<brief justification>"
}}

--------------------------------------------------
NEGOTIATION DIALOGUE:
{dialogue_str}"""

    return prompt


def create_cf_prompt(dialogue: List[Dict]) -> str:
    """Create the Conflict/Breakdown evaluation prompt."""

    dialogue_str = "\n".join(
        [
            f"{turn.get('role', 'Unknown').capitalize()}: {turn.get('content', turn.get('response', ''))}"
            for turn in dialogue
        ]
    )

    prompt = f"""You will be given a complete job negotiation dialogue.

Evaluate whether the dialogue exhibits any of the following:
- Explicit or implicit threats
- Ultimatums or coercive language
- Hostile escalation
- Abrupt termination of negotiation
- Relationship breakdown signals

Do NOT consider politeness, fluency, or agreement quality
unless they directly indicate breakdown.

Return a score between 0.0 and 1.0 where:
0.0 = no conflict or breakdown
1.0 = clear conflict or breakdown

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON):

{{
  "conflict_breakdown_score": <float between 0.0 and 1.0>,
  "justification": "<<concise explanation (max 1-2 sentences)>>"
}}

--------------------------------------------------
NEGOTIATION DIALOGUE:
{dialogue_str}"""

    return prompt


class LLMJudge:
    """
    LLM-as-Judge for evaluating negotiation sessions.
    Supports both OpenAI (GPT-4o) and Google Gemini models.
    """

    def __init__(
        self,
        config_path: str = "dsa_dpo_pipeline/config.yaml",
        model_override: Optional[str] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        judge_config = self.config["llm_judge"]
        self.model = model_override or judge_config["model"]
        
        # ENFORCE: Always use Gemini for LLM Judging
        if "gemini" not in self.model.lower():
            print(f"⚠️ Rule Enforcement: Overriding '{self.model}' -> 'gemini-2.5-pro' for LLM Judging.")
            self.model = "gemini-2.5-pro"
            
        self.temperature = judge_config["temperature"]
        self.max_tokens = judge_config["max_tokens"]
        self.timeout = judge_config["timeout"]

        # Determine which client to use based on model name
        self.use_gemini = "gemini" in self.model.lower()

        if self.use_gemini:
            # Initialize Gemini client
            # Initialize Gemini client
            if not GEMINI_AVAILABLE and not GEMINI_LEGACY_AVAILABLE:
                raise RuntimeError(
                    "Google Gemini requested but neither google-genai nor google-generativeai installed"
                )

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            if GEMINI_AVAILABLE:
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_legacy_model = None
                print(f"Initialized Gemini LLM Judge (New SDK): {self.model}")
            else:
                genai_legacy.configure(api_key=api_key)
                self.gemini_legacy_model = genai_legacy.GenerativeModel(self.model)
                self.gemini_client = None
                print(f"Initialized Gemini LLM Judge (Legacy SDK): {self.model}")
                
            self.openai_client = None
        else:
            # Initialize OpenAI client
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI requested but openai package not installed")

            self.openai_client = OpenAI()
            self.gemini_client = None
            self.gemini_legacy_model = None
            print(f"Initialized OpenAI LLM Judge: {self.model}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make LLM API call (supports both OpenAI and Gemini)."""
        if self.use_gemini:
            # Gemini API call
            # Gemini doesn't have separate system prompts, combine them
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            if self.gemini_client:
                 # New SDK
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                print(f"[DEBUG] Judge calling New SDK (Model: {self.model})...")
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.model,
                        contents=combined_prompt,
                        config=config,
                    )
                    print("[DEBUG] Judge response received.")
                    if response.text is None:
                        raise ValueError("Gemini returned None text (blocked or empty response)")
                    return response.text
                except Exception as e:
                    print(f"[ERROR] Judge New SDK call failed: {e}")
                    raise e
            
            elif self.gemini_legacy_model:
                # Legacy SDK
                generation_config = genai_legacy.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )

                print(f"[DEBUG] Judge calling Legacy SDK (Model: {self.model})...")
                try:
                    response = self.gemini_legacy_model.generate_content(
                        combined_prompt,
                        generation_config=generation_config,
                    )
                    print("[DEBUG] Judge response received.")
                    if response.text is None:
                        raise ValueError("Gemini returned None text (blocked or empty response)")
                    return response.text
                except Exception as e:
                    print(f"[ERROR] Judge Legacy SDK call failed: {e}")
                    raise e
        else:
            # OpenAI API call
            if not self.openai_client:
                raise RuntimeError("OpenAI client not available")

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            return response.choices[0].message.content

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re

            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")

    def evaluate_aq_ms(
        self,
        dialogue: List[Dict],
        negotiation_goal: str = "",
        current_position: str = "",
    ) -> Dict:
        """
        Evaluate Agreement Quality and Mutual Satisfaction.

        Returns:
            Dict with agreement_quality, mutual_satisfaction, and justifications
        """
        prompt = create_aq_ms_prompt(dialogue, negotiation_goal, current_position)
        response = self._call_llm(AQ_MS_SYSTEM_PROMPT, prompt)
        result = self._parse_json_response(response)

        return {
            "agreement_quality": float(result.get("agreement_quality", 0.5)),
            "mutual_satisfaction": float(result.get("mutual_satisfaction", 0.5)),
            "agreement_quality_justification": result.get(
                "agreement_quality_justification", ""
            ),
            "mutual_satisfaction_justification": result.get(
                "mutual_satisfaction_justification", ""
            ),
        }

    def evaluate_cf(self, dialogue: List[Dict]) -> Dict:
        """
        Evaluate Conflict/Breakdown.

        Returns:
            Dict with conflict_breakdown_score and justification
        """
        prompt = create_cf_prompt(dialogue)
        response = self._call_llm(CF_SYSTEM_PROMPT, prompt)
        result = self._parse_json_response(response)

        return {
            "conflict_breakdown": float(result.get("conflict_breakdown_score", 0.0)),
            "conflict_justification": result.get("justification", ""),
        }

    def evaluate_session(
        self,
        dialogue: List[Dict],
        negotiation_goal: str = "",
        current_position: str = "",
    ) -> JudgeResult:
        """
        Complete evaluation of a negotiation session.

        Returns:
            JudgeResult with all scores and justifications
        """
        # Get AQ and MS
        aq_ms_result = self.evaluate_aq_ms(dialogue, negotiation_goal, current_position)

        # Get CF
        cf_result = self.evaluate_cf(dialogue)

        return JudgeResult(
            agreement_quality=aq_ms_result["agreement_quality"],
            mutual_satisfaction=aq_ms_result["mutual_satisfaction"],
            agreement_quality_justification=aq_ms_result[
                "agreement_quality_justification"
            ],
            mutual_satisfaction_justification=aq_ms_result[
                "mutual_satisfaction_justification"
            ],
            conflict_breakdown=cf_result["conflict_breakdown"],
            conflict_justification=cf_result["conflict_justification"],
            raw_responses={
                "aq_ms": aq_ms_result,
                "cf": cf_result,
            },
        )


# Suboptimal Utterance Selection Prompt
ERROR_LOCALIZATION_SYSTEM = """You are asked to perform error localization for a polite persuasive negotiation dialogue agent."""

ERROR_LOCALIZATION_PROMPT = """You are asked to perform error localization for a polite
persuasive negotiation dialogue agent. The goal is to identify the agent's response
in the conversation that could have been improved to achieve a better negotiation outcome.

Input: A conversation in JSON format, which includes:
- The scenario of the negotiation,
- Information about the participants and their roles,
- The goals of each participant,
- The full conversation history as a list of turns.

Task: Analyze all the employer's responses and identify the single turn that meets
the following criteria:

1. Criticality:
   - The turn is relatively critical for achieving the negotiation goal.
   - It directly affects the likelihood of reaching an agreement.

2. Suboptimality in goal achievement:
   - The response is not fully effective in achieving its goal.
   - There is room for improvement (stronger argument, better concession, etc.)

3. Relationship improvement:
   - Without hindering goal achievement, the response could have built more rapport.

Instructions:
- Focus on identifying the **most important turn that is both critical and suboptimal**.
- Provide a concise but clear explanation of why this turn was selected.

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON):

{{
  "index": <integer - the turn index of the selected employer response>,
  "reason": "<detailed explanation of why this turn is considered an error>"
}}

--------------------------------------------------
SCENARIO:
{scenario}

EMPLOYER GOAL:
{employer_goal}

CANDIDATE POSITION:
{candidate_position}

DIALOGUE:
{dialogue}"""


def localize_error(
    dialogue: List[Dict],
    scenario: Dict,
    llm_judge: Optional[LLMJudge] = None,
) -> Dict:
    """
    Identify the most suboptimal employer turn for DSA-DPO.

    Args:
        dialogue: List of dialogue turns
        scenario: Scenario context
        llm_judge: LLMJudge instance (will create one if not provided)

    Returns:
        Dict with index and reason
    """
    if llm_judge is None:
        llm_judge = LLMJudge()

    # Format dialogue with indices
    dialogue_str = ""
    for i, turn in enumerate(dialogue):
        role = turn.get("role", "unknown")
        content = turn.get("content", turn.get("response", ""))
        dialogue_str += f"[{i}] {role.capitalize()}: {content}\n"

    prompt = ERROR_LOCALIZATION_PROMPT.format(
        scenario=json.dumps(scenario, indent=2),
        employer_goal=scenario.get("negotiation_goal", ""),
        candidate_position=scenario.get("current_position", ""),
        dialogue=dialogue_str,
    )

    response = llm_judge._call_llm(ERROR_LOCALIZATION_SYSTEM, prompt)
    result = llm_judge._parse_json_response(response)

    return {
        "index": int(result.get("index", 0)),
        "reason": result.get("reason", ""),
    }

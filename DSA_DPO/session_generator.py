"""
Session Generator for DSA-DPO Pipeline
Self-play dialogue generation with failure mode injection.
Supports both OpenAI and Google Gemini models for employer agent.
"""

import os
import json
import random
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

# Import local modules
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bc_finetuning.inference_bc import NegotiationAgent

    BC_AGENT_AVAILABLE = True
except ImportError:
    BC_AGENT_AVAILABLE = False
    print("Warning: BC agent not available. Using mock candidate.")


@dataclass
class DialogueTurn:
    """A single dialogue turn."""

    role: str  # "candidate" or "employer"
    content: str
    negotiation_strategy: str = ""
    persuasion_strategy: str = ""
    strategy_confidence: float = 1.0
    turn_index: int = 0


@dataclass
class GeneratedSession:
    """A complete generated dialogue session."""

    session_id: str
    scenario: Dict
    dialogue: List[DialogueTurn]
    failure_mode: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "scenario": self.scenario,
            "dialogue": [
                {
                    "role": t.role,
                    "content": t.content,
                    "negotiation_strategy": t.negotiation_strategy,
                    "persuasion_strategy": t.persuasion_strategy,
                    "strategy_confidence": t.strategy_confidence,
                    "turn_index": t.turn_index,
                }
                for t in self.dialogue
            ],
            "failure_mode": self.failure_mode,
            "metadata": self.metadata,
        }


class SessionGenerator:
    """
    Generates negotiation dialogue sessions via self-play.

    - Candidate: BC-trained Gemma 3 (uses learned strategies)
    - Employer: GPT-4o or Gemini with failure mode injection
    """

    def __init__(
        self,
        config_path: str = "dsa_dpo_pipeline/config.yaml",
        model_override: Optional[str] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.self_play_config = self.config["self_play"]
        self.failure_modes = self.config["failure_modes"]

        self.max_turns = self.self_play_config["max_turns"]
        self.min_turns = self.self_play_config["min_turns"]

        # Determine which model to use for employer
        self.employer_model = model_override or self.self_play_config["employer_model"]
        
        # ENFORCE: Always use OpenAI for dialogue generation, never Gemini
        if "gemini" in self.employer_model.lower():
            print(f"⚠️ Rule Enforcement: Overriding '{self.employer_model}' -> 'gpt-4o-mini' for Employer Generation.")
            self.employer_model = "gpt-4o-mini"
            
        self.use_gemini = "gemini" in self.employer_model.lower()

        # Initialize employer agent
        if self.use_gemini:
            if not GEMINI_AVAILABLE and not GEMINI_LEGACY_AVAILABLE:
                raise RuntimeError(
                    "Gemini requested but neither google-genai nor google-generativeai installed"
                )

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            if GEMINI_AVAILABLE:
                # Use new SDK
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_legacy_model = None
                print(f"Initialized Gemini employer agent (New SDK): {self.employer_model}")
            else:
                # Fallback to legacy SDK
                genai_legacy.configure(api_key=api_key)
                self.gemini_legacy_model = genai_legacy.GenerativeModel(self.employer_model)
                self.gemini_client = None
                print(f"Initialized Gemini employer agent (Legacy SDK): {self.employer_model}")
            
            self.openai_client = None
        else:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI requested but openai package not installed")

            self.openai_client = OpenAI()
            self.gemini_model = None
            print(f"Initialized OpenAI employer agent: {self.employer_model}")

        # Initialize BC agent for candidate
        self.candidate_agent = None
        if BC_AGENT_AVAILABLE:
            try:
                self.candidate_agent = NegotiationAgent(
                    base_model_path=self.self_play_config["candidate_model"],
                    adapter_path=self.self_play_config["candidate_adapter"],
                    load_in_4bit=True,
                )
            except Exception as e:
                print(f"Failed to load BC agent: {e}")

    def sample_failure_mode(self) -> Tuple[str, str]:
        """
        Sample a failure mode based on configured probabilities.

        Returns:
            Tuple of (failure_mode_name, prompt_suffix)
        """
        modes = list(self.failure_modes.keys())
        probs = [self.failure_modes[m]["probability"] for m in modes]

        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]

        selected = random.choices(modes, weights=probs, k=1)[0]
        return selected, self.failure_modes[selected]["prompt_suffix"]

    def create_employer_system_prompt(
        self,
        scenario: Dict,
        failure_suffix: str = "",
    ) -> str:
        """Create employer system prompt with optional failure injection."""

        background = scenario.get("background", "")
        employer_name = scenario.get("employer", "Employer")
        negotiation_goal = scenario.get("negotiation_goal", "")

        prompt = f"""You are an employer negotiating a job offer. Your role is "{employer_name}".

BACKGROUND: {background}

YOUR GOAL: {negotiation_goal}

INSTRUCTIONS:
- You are strictly limited to SHORT responses (approx. 15-20 words). Be concise.
- Speak naturally and conversationally, like a real human recruiter. Avoid robotic or overly formal language.
- React directly to the candidate's last point. Don't lecture.
- Be professional but pursue your negotiation objectives.
- Maintain the persona of {employer_name} fully.
{failure_suffix}

Respond with only your dialogue as the employer. Do not include any meta-commentary."""

        return prompt
        
    def create_candidate_context(self, scenario: Dict) -> str:
        """Create candidate context for BC agent."""
        current_position = scenario.get("current_position", "")
        candidate_name = scenario.get("candidate", "Candidate")

        return f"""You are a job candidate named {candidate_name}.
Current position/expectations: {current_position}
Negotiate professionally to achieve your goals."""

    def generate_employer_response(
        self,
        dialogue: List[DialogueTurn],
        scenario: Dict,
        failure_suffix: str = "",
        extra_instructions: Optional[str] = None,
    ) -> str:
        """Generate employer response using GPT-4o or Gemini.
        
        Args:
            dialogue: Conversation history
            scenario: Negotiation scenario
            failure_suffix: Failure mode injection text
            extra_instructions: Additional instructions (e.g., reinforcement for Phase 2)
        """
        system_prompt = self.create_employer_system_prompt(scenario, failure_suffix)
        
        # Add extra instructions if provided (for Phase 2 reinforcement)
        if extra_instructions:
            system_prompt = f"{system_prompt}\n\n{extra_instructions}"

        if self.use_gemini:
            # Gemini API
            # Build conversation history in Gemini format
            conversation_text = ""
            for turn in dialogue:
                speaker = "Employer" if turn.role == "employer" else "Candidate"
                conversation_text += f"{speaker}: {turn.content}\n"

            # Combine system prompt with conversation
            full_prompt = f"{system_prompt}\n\nCONVERSATION SO FAR:\n{conversation_text}\nEmployer:"

            if self.gemini_client:
                # New SDK Usage
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                )
                
                print(f"[DEBUG] Generating content with New SDK (Model: {self.employer_model})...")
                print(f"[DEBUG] Prompt length: {len(full_prompt)}")
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.employer_model,
                        contents=full_prompt,
                        config=config
                    )
                    print("[DEBUG] Content generated successfully.")
                    return response.text.strip()
                except Exception as e:
                    print(f"[ERROR] New SDK generation failed: {e}")
                    raise e
            
            elif self.gemini_legacy_model:
                # Legacy SDK Usage
                generation_config = genai_legacy.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                )

                print(f"[DEBUG] Generating content with Legacy SDK (Model: {self.employer_model})...")
                try:
                    response = self.gemini_legacy_model.generate_content(
                        full_prompt,
                        generation_config=generation_config,
                    )
                    print("[DEBUG] Content generated successfully.")
                    return response.text.strip()
                except Exception as e:
                    print(f"[ERROR] Legacy SDK generation failed: {e}")
                    raise e
        else:
            # OpenAI API
            if not self.openai_client:
                return "I appreciate your perspective. Let me consider that."

            # Build conversation history
            messages = [{"role": "system", "content": system_prompt}]

            for turn in dialogue:
                role = "assistant" if turn.role == "employer" else "user"
                messages.append({"role": role, "content": turn.content})

            response = self.openai_client.chat.completions.create(
                model=self.employer_model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )

            return response.choices[0].message.content.strip()

    def generate_candidate_response(
        self,
        dialogue: List[DialogueTurn],
        scenario: Dict,
    ) -> Tuple[str, Dict]:
        """
        Generate candidate response using BC-trained agent.

        Returns:
            Tuple of (response_text, strategy_info)
        """
        if self.candidate_agent is None:
            # Fallback mock response
            responses = [
                "I understand your position. Could we discuss the compensation package?",
                "That's an interesting offer. What about the benefits?",
                "I appreciate the flexibility. My expectation is slightly higher.",
                "Thank you for explaining. I'd like to negotiate the salary further.",
            ]
            return random.choice(responses), {
                "negotiation_strategy": "Collaborative Style"
            }

        # Build history for BC agent
        history = [{"role": t.role, "content": t.content} for t in dialogue]

        # Get last employer utterance (if any)
        if dialogue and dialogue[-1].role == "employer":
            employer_utterance = dialogue[-1].content
        else:
            employer_utterance = "Let's discuss the position."

        # Generate with BC agent
        result = self.candidate_agent.respond(
            candidate_utterance=employer_utterance,  # BC agent responds to last turn
            dialogue_history=history,
            return_full_output=True,
        )

        strategy_info = {}
        if result.candidate_strategy:
            strategy_info["negotiation_strategy"] = (
                result.candidate_strategy.negotiation_strategy
            )
            strategy_info["persuasion_strategy"] = (
                result.candidate_strategy.persuasion_strategy
            )

        return result.response, strategy_info

    def generate_session(
        self,
        scenario: Dict,
        force_failure_mode: Optional[str] = None,
    ) -> GeneratedSession:
        """
        Generate a complete dialogue session.

        Args:
            scenario: Scenario configuration
            force_failure_mode: Override random failure mode selection

        Returns:
            GeneratedSession with complete dialogue
        """
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        # Sample failure mode
        if force_failure_mode:
            failure_mode = force_failure_mode
            failure_suffix = self.failure_modes.get(failure_mode, {}).get(
                "prompt_suffix", ""
            )
        else:
            failure_mode, failure_suffix = self.sample_failure_mode()

        dialogue: List[DialogueTurn] = []
        turn_index = 0

        # Generate initial employer greeting
        initial_employer = self.generate_employer_response([], scenario, failure_suffix)
        dialogue.append(
            DialogueTurn(
                role="employer",
                content=initial_employer,
                turn_index=turn_index,
            )
        )
        turn_index += 1

        # Self-play loop
        while turn_index < self.max_turns:
            # Candidate turn
            candidate_response, candidate_strategy = self.generate_candidate_response(
                dialogue, scenario
            )
            dialogue.append(
                DialogueTurn(
                    role="candidate",
                    content=candidate_response,
                    negotiation_strategy=candidate_strategy.get(
                        "negotiation_strategy", ""
                    ),
                    persuasion_strategy=candidate_strategy.get(
                        "persuasion_strategy", ""
                    ),
                    turn_index=turn_index,
                )
            )
            turn_index += 1

            # Check for natural ending
            if self._check_dialogue_end(dialogue):
                break

            if turn_index >= self.max_turns:
                break

            # Employer turn
            employer_response = self.generate_employer_response(
                dialogue, scenario, failure_suffix
            )
            dialogue.append(
                DialogueTurn(
                    role="employer",
                    content=employer_response,
                    turn_index=turn_index,
                )
            )
            turn_index += 1

            # Check for natural ending
            if self._check_dialogue_end(dialogue):
                break

        return GeneratedSession(
            session_id=session_id,
            scenario=scenario,
            dialogue=dialogue,
            failure_mode=failure_mode,
            metadata={
                "total_turns": len(dialogue),
                "failure_suffix_used": failure_suffix,
            },
        )

    def _check_dialogue_end(self, dialogue: List[DialogueTurn]) -> bool:
        """Check if dialogue has reached a natural conclusion."""
        if len(dialogue) < self.min_turns:
            return False

        last_content = dialogue[-1].content.lower()

        # Check for agreement/conclusion signals
        end_signals = [
            "accept",
            "agreed",
            "deal",
            "look forward",
            "welcome aboard",
            "offer letter",
            "thank you for the opportunity",
            "final offer",
            "cannot proceed",
            "unfortunately",
        ]

        return any(signal in last_content for signal in end_signals)

    def generate_batch(
        self,
        scenarios: List[Dict],
        batch_size: int = 10,
    ) -> List[GeneratedSession]:
        """Generate a batch of sessions from scenarios."""
        sessions = []

        for i in range(min(batch_size, len(scenarios))):
            scenario = scenarios[i % len(scenarios)]
            session = self.generate_session(scenario)
            sessions.append(session)

        return sessions


def load_scenarios(dataset_path: str) -> List[Dict]:
    """Load scenarios from the HR dataset (conversations file)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = []
    for conv in data:
        scenario = conv.get("original_scenario", {})
        scenario["scenario_id"] = conv.get("scenario_id", "")
        scenarios.append(scenario)

    return scenarios


def load_scenarios_from_dir(scenario_dir: str) -> List[Dict]:
    """Load scenarios directly from a directory of scenario JSON files.

    Each JSON file should contain a list of scenario dicts with keys:
    domain, background, employer, candidate, negotiation_goal, current_position.

    Args:
        scenario_dir: Path to directory containing scenario JSON files.

    Returns:
        Flat list of all scenario dicts across all files.
    """
    import glob

    scenarios = []
    json_files = sorted(glob.glob(os.path.join(scenario_dir, "*.json")))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {scenario_dir}")

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list-of-scenarios and single-scenario files
        if isinstance(data, list):
            for idx, sc in enumerate(data):
                sc.setdefault("scenario_id", f"{os.path.basename(fpath)}_{idx}")
                scenarios.append(sc)
        elif isinstance(data, dict):
            data.setdefault("scenario_id", os.path.basename(fpath))
            scenarios.append(data)

    return scenarios


def main():
    """Test session generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate DSA-DPO sessions")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../hr_conversations/final_hr_conversations_dataset_fixed.json",
    )
    parser.add_argument(
        "--output", type=str, default="dsa_dpo_pipeline/outputs/test_sessions.json"
    )
    parser.add_argument("--count", type=int, default=5)

    args = parser.parse_args()

    print("Loading scenarios...")
    scenarios = load_scenarios(args.dataset)
    print(f"Loaded {len(scenarios)} scenarios")

    print("Initializing generator...")
    generator = SessionGenerator()

    print(f"Generating {args.count} sessions...")
    sessions = generator.generate_batch(scenarios, batch_size=args.count)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sessions], f, indent=2)

    print(f"Saved {len(sessions)} sessions to {args.output}")

    # Print sample
    if sessions:
        print("\n--- Sample Session ---")
        sample = sessions[0]
        print(f"Session ID: {sample.session_id}")
        print(f"Failure Mode: {sample.failure_mode}")
        for turn in sample.dialogue[:4]:
            print(f"{turn.role.capitalize()}: {turn.content[:100]}...")


if __name__ == "__main__":
    main()

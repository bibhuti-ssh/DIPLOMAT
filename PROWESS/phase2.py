import json
import os
import time
import re
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()

INPUT_DIR = "workplace_scenarios_json"
OUTPUT_DIR = "workplace_conversations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
total_mini_input_tokens = 0
total_mini_output_tokens = 0
total_o_input_tokens = 0
total_o_output_tokens = 0

MINI_INPUT_COST_PER_1K = 0.000150
MINI_OUTPUT_COST_PER_1K = 0.000600


O_INPUT_COST_PER_1K = 0.005000 
O_OUTPUT_COST_PER_1K = 0.015000 

PERSUASION_STRATEGIES_DATA = [
    {"Strategy": "Rapport Building", "Definition": "Establishing trust and a positive relationship to enhance receptiveness."},
    {"Strategy": "Concern Addressing", "Definition": "Actively listening to and resolving objections or hesitations."},
    {"Strategy": "Emotional Appeal", "Definition": "Connecting through emotions such as empathy, excitement, or urgency."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
    {"Strategy": "Credibility & Confidence", "Definition": "Demonstrating expertise and confidence to increase trust and influence."},
    {"Strategy": "Data-Driven Persuasion", "Definition": "Using evidence, facts, and benchmarks to strengthen arguments."},
    {"Strategy": "Problem-Solving Focus", "Definition": "Presenting ideas as solutions that address mutual challenges."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
    {"Strategy": "Self-Interest Appeal", "Definition": "Framing arguments around how the outcome directly benefits the other party."},
    {"Strategy": "Value Alignment", "Definition": "Linking your proposal to the other party's core values and principles."},
    {"Strategy": "Reputation Highlighting", "Definition": "Leveraging past achievements or organizational standing to reinforce trustworthiness."},
    {"Strategy": "Future Vision Alignment", "Definition": "Connecting the proposal with shared long-term goals and aspirations."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
]

NEGOTIATION_STRATEGIES_DATA = [
    {"Strategy": "Collaborative Style", "Definition": "Working jointly to find mutually beneficial outcomes and preserve relationships."},
    {"Strategy": "Active Listening", "Definition": "Demonstrating attentiveness and understanding to build rapport and trust."},
    {"Strategy": "Win-Win Framing", "Definition": "Framing negotiation as a shared problem to solve for mutual benefit."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
    {"Strategy": "Principled Negotiation", "Definition": "Focusing on mutual interests and objective standards rather than positions."},
    {"Strategy": "Data-Driven Justification", "Definition": "Supporting negotiation points with evidence like market benchmarks and past performance."},
    {"Strategy": "MESO (Multiple Equivalent Simultaneous Offers)", "Definition": "Proposing multiple offers of equal value to reveal priorities and increase agreement likelihood."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
    {"Strategy": "Anchoring", "Definition": "Setting a strong initial offer to influence the negotiation range."},
    {"Strategy": "Door-in-the-Face", "Definition": "Starting with a larger request to make the actual target seem more acceptable."},
    {"Strategy": "Reciprocal Concessions", "Definition": "Offering small concessions to encourage reciprocation from the other party."},
    {"Strategy": "Credibility Assertion", "Definition": "Building trust by reinforcing personal or organizational credibility during negotiations."},
    {"Strategy": "No Strategy", "Definition": "Indicates the absence of any explicit persuasion strategy."},
]


def extract_wait_time(error_message: str) -> int:
    """Extract wait time from error message"""
    patterns = [
        r'after\s+(\d+)\s*seconds',
        r'after\s+(\d+)\s*s',
        r'wait\s+(\d+)\s*seconds',
        r'wait\s+(\d+)\s*s',
        r'retry\s+after\s+(\d+)\s*seconds',
        r'retry\s+after\s+(\d+)\s*s'
    ]
    for pattern in patterns:
        match = re.search(pattern, error_message.lower())
        if match:
            return int(match.group(1))
    return 30


def call_llm(prompt: str, system_prompt: str = "", max_retries: int = 3) -> str:
    """Make LLM call with retry logic"""
    global total_mini_input_tokens, total_mini_output_tokens

    full_prompt = system_prompt + "\n\n" + prompt if system_prompt else prompt
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=8192
            )

            total_mini_input_tokens += response.usage.prompt_tokens
            total_mini_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            if any(k in error_str for k in ["rate", "429", "quota", "retry"]):
                wait_time = extract_wait_time(error_msg)
                print(f"⚠️ Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ LLM error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
    return ""


def call_llm2(prompt: str, system_prompt: str = "", max_retries: int = 3) -> str:
    """Make LLM call with retry logic"""
    global total_o_input_tokens, total_o_output_tokens

    full_prompt = system_prompt + "\n\n" + prompt if system_prompt else prompt
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=16384
            )

            total_o_input_tokens += response.usage.prompt_tokens
            total_o_output_tokens += response.usage.completion_tokens

            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            if any(k in error_str for k in ["rate", "429", "quota", "retry"]):
                wait_time = extract_wait_time(error_msg)
                print(f"⚠️ Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ LLM error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
    return ""


def clean_json_response(response: str) -> str:
    """Clean and extract JSON from response"""
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    start_idx = cleaned.find('[')
    end_idx = cleaned.rfind(']') + 1
    if start_idx >= 0 and end_idx > start_idx:
        return cleaned[start_idx:end_idx]
    return cleaned


def generate_conversation(scenario: Dict, strategy_bucket: Dict) -> List[Dict]:
    # system_prompt = """You are an expert dialogue writer specializing in professional workplace conversations..."""
    system_prompt = """You are an expert dialogue writer specializing in professional workplace conversations. You ace in generating different types of conversation simulations uniquely for similar scenarios.
    Generate realistic, engaging dialogues that demonstrate various workplace persuasion techniques in professional settings.
    You must return a valid JSON array of dialogue objects. Each dialogue object must have 'role' and 'response' fields."""
    
    prompt = f"""
    Generate a professional Workplace persuasion dialogue for this scenario.

    # SCENARIO:
    {json.dumps(scenario, indent=2)}

    # AVAILABLE STRATEGIES:
    - Persuasion Strategies: {json.dumps([{"Strategy": s["Strategy"], "Definition": s["Definition"]} for s in strategy_bucket["persuasion"]], indent=2)}
    - Negotiation Strategies: {json.dumps([{"Strategy": s["Strategy"], "Definition": s["Definition"]} for s in strategy_bucket["negotiation"]], indent=2)}

    # REQUIREMENTS:
    1. There should be minimum of 18 turns to 22 turns of dialogues
    2. Professional workplace tone throughout
    3. Realistic back-and-forth conversation
    4. Show persuasion progression naturally
    5. Include realistic objections and responses
    6. End with clear resolution

    # RULES FOR STRATEGY SELECTION:
    1. For each employer's turn:
       - Analyze the candidate's most recent response
       - Select appropriate strategies from available lists
       - Generate natural dialogue incorporating strategies

    # RULES FOR EMPLOYER:
    0. Dialogues should appear natural hence 20-25 words in length
    1. Possess rich persuasion experience with politeness
    2. Initiate the first round of dialogue
    3. Words should be persuasive and penetrating
    4. Pay attention to key time points in speech

    # RULES FOR CANDIDATE:
    0. Dialogues should appear natural hence 20-25 words in length
    1. Guide the employer to deliver excellent persuasive speech
    2. Show realistic reactions: hesitation, false commitment, impatience
    3. Maintain professional tone

    # OUTPUT FORMAT - JSON ONLY:
    [
        {{
            "role": "employer",
            "response": "dialogue text"
        }},
        {{
            "role": "candidate", 
            "response": "dialogue text"
        }},
        ...
    ]

    CRITICAL: Exactly 18-22 dialogue turns. Return ONLY valid JSON array.
    """


    try:
        response = call_llm(prompt, system_prompt)
        if not response:
            return []
        cleaned_response = clean_json_response(response)
        conversation = json.loads(cleaned_response)
        if 18 <= len(conversation) <= 30:
            return conversation
        else:
            print(f"⚠️ Conversation length {len(conversation)} outside range 18-30")
            return []
    except Exception as e:
        print(f"❌ Conversation generation failed: {e}")
        return []
    
def label_unlabeled_strategies(conversation: List[Dict], strategy_map: Dict) -> List[Dict]:
    # Step 1: Find unlabeled turns (missing values)
    unlabeled_turns = [
        idx for idx, turn in enumerate(conversation)
        if not turn.get("negotiation_strategy") and not turn.get("persuasion_strategy")
    ]
    if not unlabeled_turns:
        print("✅ No unlabeled turns remaining.")
        return conversation

    # Step 2: Prepare prompt
    system_prompt = """
    You are an expert HR strategy analyst. Analyze the conversation and label ONLY the given turns.
    """

    prompt = f"""
    # FULL CONVERSATION (original):
    {json.dumps([{'role': t['role'], 'response': t['response']} for t in conversation], indent=2)}

    # TURNS TO LABEL:
    {unlabeled_turns}
"leave_policy.json", "performance_review.json", "salary_negotiation.json", "training_opportunities.jso"
    # AVAILABLE STRATEGIES:
    Persuasion: {[s["Strategy"] for s in PERSUASION_STRATEGIES_DATA]}
    Negotiation: {[s["Strategy"] for s in NEGOTIATION_STRATEGIES_DATA]}

    # TASK:
    For EACH UNLABELED turn in the conversation, analyze and provide:
    1. ONE most relevant persuasion strategy (just the name) strictly among the available "Persuasion" strategies
    2. ONE most relevant negotiation strategy (just the name) strictly among the available "Negotiation" strategies
    3. Brief reasoning for each choice (Maximum of 2 sentences explaining why this strategy fits this specific turn in context of the entire conversation)
    # CRITICAL : 
    - Make sure you Label all given UNLABELED turns
    # REQUIRED FIELDS:
    - turn_index
    - role
    - negotiation_strategy
    - negotiation_strategy_reasoning (max 2-3 sentences)
    - persuasion_strategy
    - persuasion_strategy_reasoning (max 2-3 sentences)

    # OUTPUT FORMAT (JSON ONLY):
    [
        {{
            "turn_index": 0,
            "role": "employer",
            "negotiation_strategy": "...",
            "negotiation_strategy_reasoning": "...",
            "persuasion_strategy": "...",
            "persuasion_strategy_reasoning": "..."
        }}
    ]
    """

    try:
        response = call_llm2(prompt, system_prompt)
        cleaned = clean_json_response(response)
        new_labels = json.loads(cleaned)

        for item in new_labels:
            idx = item["turn_index"]
            conversation[idx].update(item)

        print(f"✅ Labeled {len(new_labels)} previously unlabeled turns.")
        for idx, turn in enumerate(conversation):
            turn["turn_index"] = idx

        return conversation

    except Exception as e:
        for idx, turn in enumerate(conversation):
            turn["turn_index"] = idx
        print(f"❌ Relabeling failed: {e}")
        return conversation




def label_strategies(conversation: List[Dict], strategy_bucket: Dict) -> List[Dict]:
    system_prompt = """You are an expert HR strategy analyst. Analyze the entire conversation and identify persuasion and negotiation strategies used by BOTH employer and candidate in each of their turns."""
    
    prompt = f"""
    Analyze this entire HR conversation and identify strategies used by BOTH employer AND candidate in each of their turns.

    # FULL CONVERSATION:
    {json.dumps(conversation, indent=2)}

    # AVAILABLE STRATEGIES:
    Persuasion Strategies: {[s["Strategy"] for s in strategy_bucket["persuasion"]]}
    Negotiation Strategies: {[s["Strategy"] for s in strategy_bucket["negotiation"]]}

    # TASK:
    For EACH turn in the conversation (both employer AND candidate), analyze and provide:
    1. ONE most relevant persuasion strategy (just the name) strictly among the available "Persuasion" strategies
    2. ONE most relevant negotiation strategy (just the name) strictly among the available "Negotiation" strategies
    3. Brief reasoning for each choice (Maximum of 2 sentences explaining why this strategy fits this specific turn in context of the entire conversation)

    # OUTPUT FORMAT - JSON ONLY:
    [
        {{
            "turn_index": 0,
            "role": "employer",
            "negotiation_strategy": "selected strategy name",
            "negotiation_strategy_reasoning": "detailed reasoning why this strategy fits this specific turn",
            "persuasion_strategy": "selected strategy name",
            "persuasion_strategy_reasoning": "detailed reasoning why this strategy fits this specific turn"
        }},
        {{
            "turn_index": 1, 
            "role": "candidate",
            "negotiation_strategy": "selected strategy name",
            "negotiation_strategy_reasoning": "detailed reasoning why this strategy fits this specific turn",
            "persuasion_strategy": "selected strategy name",
            "persuasion_strategy_reasoning": "detailed reasoning why this strategy fits this specific turn"
        }},
        ... for ALL turns in the conversation
    ]

    Important:
    - Analyze ALL turns (both employer AND candidate) in the context of the entire conversation flow
    - Provide specific, contextual reasoning for each strategy choice
    - Ensure ALL turns are analyzed (turn_index should match their position in conversation)
    - Both parties can use persuasion and negotiation strategies
    - Return ONLY the GIVEN EXAMPLE JSON array without any additional text and perfect parameter fields.
    """

    try:
        response = call_llm2(prompt, system_prompt)
        if response:
            cleaned = clean_json_response(response)
            analysis_data = json.loads(cleaned)
            print("\n here--- \n ")
            print(analysis_data)
            print(f"🔍 Received strategy analysis for {len(analysis_data)} turns")

            strategy_map = {}
            for analysis in analysis_data:
                turn_idx = analysis.get("turn_index", -1)
                if turn_idx >= 0:
                    strategy_map[turn_idx] = {
                        "negotiation_strategy": analysis.get("negotiation_strategy", ""),
                        "negotiation_strategy_reasoning": analysis.get("negotiation_strategy_reasoning", ""),
                        "persuasion_strategy": analysis.get("persuasion_strategy", ""),
                        "persuasion_strategy_reasoning": analysis.get("persuasion_strategy_reasoning", "")
                    }

            labeled_conversation = []
            cnt = 0 
            for i, turn in enumerate(conversation):
                if i in strategy_map:
                    turn.update(strategy_map[i])
                    print(f"✅ Labeled turn {i+1} ({turn['role']}) with strategies")
                else:
                    cnt+= 1
                    turn.update({
                        "negotiation_strategy": "",
                        "negotiation_strategy_reasoning": "",
                        "persuasion_strategy": "",
                        "persuasion_strategy_reasoning": ""
                    })
                    print(f"⚠️ No strategy analysis for turn {i+1}")
                labeled_conversation.append(turn)

            if cnt > 0: 
                print(f"⚠️ {cnt} turns were not labeled. Retrying unlabeled turns...")
                return label_unlabeled_strategies(labeled_conversation, strategy_map)
            else :
                print(f"!! No unlabeled turns to process")
                print(f"✅ Successfully labeled {len(strategy_map)}/{len(conversation)} turns")
                for idx, turn in enumerate(labeled_conversation):
                    turn["turn_index"] = idx                
                return labeled_conversation

    except Exception as e:
        print(f"❌ Strategy labeling failed: {e}")
        return conversation


def main(ip_file: str):
    global total_input_tokens, total_output_tokens
    print(f"🔄 Loading scenario file {ip_file}...")

    all_scenarios = []
    filepath = os.path.join(INPUT_DIR, ip_file)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
            if isinstance(scenarios, list):
                all_scenarios.extend(scenarios)
    except Exception as e:
        print("❌ Cannot open the file:", e)

    if not all_scenarios:
        print("❌ No scenarios found!")
        return

    print(f"📊 Total scenarios to process: {len(all_scenarios)}")

    strategy_buckets = [
        {
            "name": "bucket_1",
            "persuasion": PERSUASION_STRATEGIES_DATA[:4],  
            "negotiation": NEGOTIATION_STRATEGIES_DATA[:4]
        },
        {
            "name": "bucket_2", 
            "persuasion": PERSUASION_STRATEGIES_DATA[4:8],  
            "negotiation": NEGOTIATION_STRATEGIES_DATA[4:8]
        },
        {
            "name": "bucket_3",
            "persuasion": PERSUASION_STRATEGIES_DATA[8:],   
            "negotiation": NEGOTIATION_STRATEGIES_DATA[8:]
        }
    ]

    all_conversations = []

    for scenario_idx, scenario in enumerate(all_scenarios):
        print(f"\n{'='*60}")
        print(f"🎯 Processing Scenario {scenario_idx+1}/{len(all_scenarios)}")
        print(f"Domain: {scenario.get('domain', 'Unknown')}")
        print(f"{'='*60}")
        
        for bucket_idx, bucket in enumerate(strategy_buckets):
            print(f"\n🪣 Using Strategy Bucket {bucket_idx+1}: {bucket['name']}")
            
            # Generate conversation
            for attempt in range(2) : 
                print("here is the \n ", attempt)
                conversation = generate_conversation(scenario, bucket)
                if not conversation:
                    print("❌ Conversation generation failed, skipping...")
                    continue
                print("\n CONVO:: \n")
                print(conversation)
                # Label strategies
                labeled_conversation = label_strategies(conversation, bucket)
                print("\n LABELED CONVO:: \n")
                print(labeled_conversation)
                # Create final output
                conversation_data = {
                    "scenario_id": f"scenario_{scenario_idx+1}",
                    "bucket_id": bucket['name'],
                    "original_scenario": scenario,
                    "strategies_used": {
                        "persuasion": [s["Strategy"] for s in bucket["persuasion"]],
                        "negotiation": [s["Strategy"] for s in bucket["negotiation"]]
                    },
                    "conversation": labeled_conversation,
                    "total_turns": len(labeled_conversation),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                all_conversations.append(conversation_data)
            print(f"✅ Successfully processed with bucket {bucket_idx+1}")


            with open(os.path.join(OUTPUT_DIR, f"all_conversations_{ip_file}.json"), "w", encoding="utf-8") as f:
                json.dump(all_conversations, f, indent=2, ensure_ascii=False)
            
        
            time.sleep(5)
        
        
        print("⏳ Waiting 10 seconds before next scenario...")
        time.sleep(10)
    
    
    print(f"\n💾 Saving final output: {len(all_conversations)} conversations")
    with open(os.path.join(OUTPUT_DIR, f"final_workplace_conversations_dataset_{ip_file}.json"), "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)

    total_cost = (total_mini_input_tokens / 1000 * MINI_INPUT_COST_PER_1K) + \
                    (total_mini_output_tokens / 1000 * MINI_OUTPUT_COST_PER_1K) + \
                    (total_o_input_tokens / 1000 * MINI_INPUT_COST_PER_1K) + \
                    (total_o_output_tokens / 1000 * MINI_OUTPUT_COST_PER_1K)

    print("\n🎉 Pipeline completed successfully!")
    print(f"📈 Total conversations generated: {len(all_conversations)}")
    print(f"🧾 Token usage summary:")
    print(f"   - Grand Total Tokens: {total_mini_input_tokens + total_mini_output_tokens + total_o_input_tokens + total_o_output_tokens}")
    print(f"💰 Estimated Total Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    ip_files = ["hiring_decision.json" ]
    for ip_file in ip_files : 
        main(ip_file)
        print(" \n\n ========================= Next File ========================= \n\n ")
    
# PROWESS: A Workplace Negotiation Dialogue Dataset

> **Note:** This repository contains the first **600 samples** from the complete PROWESS dataset (2,400 total dialogues).

## Overview

**PROWESS** is a novel dialogue dataset designed for workplace negotiation scenario research. It consists of 2,400 multi-turn interactions between a candidate and an employer, covering key negotiation dimensions such as compensation, working conditions, career progression, and related employment terms. PROWESS spans **25 common workplace negotiation aspects**, reflecting realistic discussions that arise during hiring and professional interactions in the workplace.

## Dataset Construction Pipeline

The construction of PROWESS involves three essential steps:

1. **Negotiation, Persuasion, and Politeness Annotation Schema Designing**
2. **Multi-agent Dialogue Generation**
3. **Dataset Filtering and Quality Assessment**

---

## Negotiation Dimensions

PROWESS covers 25 workplace negotiation aspects across various professional contexts:

- Hiring Decision
- Onboarding
- Performance Review
- Remote Work Arrangement
- Benefits Negotiation
- Contract Terms
- Relocation Package
- Flexible Hours
- Career Development
- Training Opportunities
- Bonus Structure
- Equity Negotiation
- Vacation Time
- Work-Life Balance
- Project Assignment
- Team Transition
- Exit Interview
- Retention Offer
- Job Responsibilities
- Work Equipment
- Professional Development
- Health Benefits
- Retirement Plans

---

## Annotation Schema

### Negotiation Strategies (11 Strategies)

Workplace negotiation relies on information sharing, transparency, and relational rapport to support collaboration and professional growth. We define a set of 11 negotiation strategies grounded in workplace negotiation principles, designed for problem-solving, concession-making, position-establishing, or relationship-building:

1. **Collaborative Style** - Working jointly to find mutually beneficial outcomes and preserve relationships
2. **Active Listening** - Demonstrating attentiveness and understanding to build rapport and trust
3. **Win-Win Framing** - Framing negotiation as a shared problem to solve for mutual benefit
4. **Principled Negotiation** - Focusing on mutual interests and objective standards rather than positions
5. **Data-Driven Justification** - Supporting negotiation points with evidence like market benchmarks and past performance
6. **MESO (Multiple Equivalent Simultaneous Offers)** - Proposing multiple offers of equal value to reveal priorities and increase agreement likelihood
7. **Anchoring** - Setting a strong initial offer to influence the negotiation range
8. **Door-in-the-Face** - Starting with a larger request to make the actual target seem more acceptable
9. **Reciprocal Concessions** - Offering small concessions to encourage reciprocation from the other party
10. **Credibility Assertion** - Building trust by reinforcing personal or organizational credibility during negotiations
11. **No Strategy** - Indicates the absence of any explicit negotiation strategy

### Persuasion Strategies (11 Strategies)

Persuasion complements negotiation strategies by guiding negotiation dynamics, influencing decisions, and shaping relationships. We define 11 persuasion strategies guided by persuasive negotiation theory:

1. **Rapport Building** - Establishing trust and a positive relationship to enhance receptiveness
2. **Concern Addressing** - Actively listening to and resolving objections or hesitations
3. **Emotional Appeal** - Connecting through emotions such as empathy, excitement, or urgency
4. **Credibility & Confidence** - Demonstrating expertise and confidence to increase trust and influence
5. **Data-Driven Persuasion** - Using evidence, facts, and benchmarks to strengthen arguments
6. **Problem-Solving Focus** - Presenting ideas as solutions that address mutual challenges
7. **Self-Interest Appeal** - Framing arguments around how the outcome directly benefits the other party
8. **Value Alignment** - Linking your proposal to the other party's core values and principles
9. **Reputation Highlighting** - Leveraging past achievements or organizational standing to reinforce trustworthiness
10. **Future Vision Alignment** - Connecting the proposal with shared long-term goals and aspirations
11. **No Strategy** - Indicates the absence of any explicit persuasion strategy

### Politeness Levels (4 Levels)

Politeness influences negotiation outcomes by mitigating conflict and promoting sociopsychological closeness. We adopt four politeness levels from workplace negotiation practice:

1. **Low Polite** - Minimal courtesy; direct and assertive communication
2. **Moderate Polite** - Balanced professionalism with standard courtesies
3. **High Polite** - Extensive politeness markers and deferential language
4. Standard polite conventions in professional settings

---

## Dataset Filtering and Quality Assessment

### Filtering Process

Once the complete dataset is generated, comprehensive inspection removes erroneous dialogues. We identify and remove five types of issues:

1. **Empty Utterances** - Turns with no meaningful content
2. **Repetitive Utterances** - Redundant or copy-pasted dialogue
3. **Insufficient Interaction Rounds** - Conversations with too few turns
4. **Incomplete or Missing Annotations** - Turns lacking required strategy labels
5. **Improper Conversation Openings or Closings** - Dialogues with unrealistic beginnings or endings

### Human Quality Evaluation

The remaining dialogues are qualitatively evaluated by three human evaluators (Ph.D. holders in Linguistics with expertise in negotiation, persuasion, and politeness concepts) under the supervision of a domain expert in business management. Each dialogue is rated on the following 8 dimensions using a 1–5 scale (low to high):

| Metric | Description |
|--------|-------------|
| **SC** (Scenario Consistency) | Dialogue alignment with the specified workplace scenario |
| **NSC** (Negotiation Strategy Correctness) | Appropriate and correct application of negotiation strategies |
| **PSC** (Persuasion Strategy Correctness) | Appropriate and correct application of persuasion strategies |
| **PA** (Politeness Appropriateness) | Suitable politeness levels for the context |
| **F** (Fairness) | Balanced treatment of both parties' interests |
| **C** (Coherence) | Logical flow and consistency throughout dialogue |
| **N** (Naturalness) | Realistic and natural-sounding language |
| **E** (Engagingness) | Interest and engagement level of the dialogue |

Only dialogues with scores ≥ 3 across all metrics are retained.

### Evaluation Results

The retained 2,400 dialogues achieve the following average ratings:

| Metric | Average Rating | Inter-Evaluator κ Score |
|--------|----------------|------------------------|
| SC | 4.58 | 0.84 |
| NSC | 4.46 | 0.82 |
| PSC | 4.39 | 0.80 |
| PA | 4.51 | 0.83 |
| F | 4.32 | 0.79 |
| C | 4.37 | 0.81 |
| N | 4.79 | 0.85 |
| E | 4.41 | 0.80 |

The κ values indicate **substantial to near-perfect inter-evaluator agreement**, confirming the reliability and consistency of human evaluations. Results demonstrate that the retained dialogues exhibit:

- Strong scenario grounding
- Consistent use of negotiation and persuasion strategies
- Appropriate politeness levels
- High levels of fairness, coherence, naturalness, and engagement

---

## Dataset Structure

### Main Files

- **PROWESS.json** - Final consolidated dataset containing all 2,400 filtered and validated workplace negotiation dialogues
- **phase1.py** - Scenario generation pipeline (generates initial workplace negotiation scenarios)
- **phase2.py** - Dialogue generation and strategy labeling pipeline (generates multi-turn conversations with strategy annotations)
- **strategy_validation.py** - Validation script for strategy annotations and data quality

### Generated Directories

The generation pipeline creates the following output directories:

- **workplace_scenarios_json/** - Generated workplace negotiation scenarios (25 domains)
- **workplace_conversations/** - Generated multi-turn dialogues with strategy labels

---

## Generation Pipeline

### Phase 1: Scenario Generation
`phase1.py` generates initial workplace negotiation scenarios for each of the 25 negotiation domains using LLMs.

**Key Features:**
- Generates 16 diverse scenarios per domain
- Ensures scenario uniqueness across company sizes, industries, and job roles
- Creates explicit employer and candidate roles
- Includes workplace context and clear negotiation objectives

**Output:** JSON files with scenario descriptions for each domain

### Phase 2: Dialogue Generation & Strategy Labeling
`phase2.py` generates realistic multi-turn dialogues from scenarios and applies strategy labels.

**Key Features:**
- Generates 18-22 turn dialogues for each scenario
- Creates 3 dialogue variations per scenario using different strategy buckets
- Applies negotiation and persuasion strategy labels to each turn
- Ensures natural language with professional tone (20-25 words per turn)

**Output:** Complete dialogues with turn-level strategy annotations

---

## Data Format

### Dialogue Structure

Each dialogue in the dataset includes:

```json
{
  "scenario_id": "scenario_1",
  "original_scenario": {
    "domain": "hiring decision",
    "background": "Scenario description",
    "employer": "HR Manager",
    "candidate": "Software Engineer",
    "negotiation_goal": "Negotiate salary",
    "current_position": "Initial candidate position"
  },
  "strategies_used": {
    "persuasion": ["Strategy 1", "Strategy 2", ...],
    "negotiation": ["Strategy 1", "Strategy 2", ...]
  },
  "conversation": [
    {
      "turn_index": 0,
      "role": "employer",
      "response": "Dialogue text",
      "negotiation_strategy": "Strategy name",
      "negotiation_strategy_reasoning": "Why this strategy...",
      "persuasion_strategy": "Strategy name",
      "persuasion_strategy_reasoning": "Why this strategy..."
    },
    ...
  ],
  "total_turns": 20,
  "generated_at": "2024-01-01 12:00:00"
}
```

---

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| **Negotiation Aspects** | 25 |
| **Total Dialogues** | 2,400 |
| **Total Utterances** | 51,796 |
| **Total Words** | 941,900 |
| **Avg. Utterances per Dialogue** | 21.58 |
| **Minimum Dialogue Length** | 18 |
| **Maximum Dialogue Length** | 26 |
| **Avg. Words per Dialogue** | 392.5 |
| **Avg. Words per Utterance** | 18.2 |
| **Inter-Evaluator Agreement** | κ = 0.80-0.85 (substantial to near-perfect) |

---

## Notes

- The dataset is designed for training and evaluating dialogue systems in workplace negotiation contexts
- All dialogues are synthetic but grounded in realistic workplace scenarios
- Strategy labels are assigned at the turn level for fine-grained analysis
- The dataset emphasizes practical negotiation strategies applicable in professional settings
- All workplace terminology and conventions follow modern professional standards

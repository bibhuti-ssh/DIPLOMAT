
import json
import os
import time
import asyncio
import re
import google.generativeai as genai
from typing import List, Dict


SCENARIOS_PER_DOMAIN = 16
DOMAINS = [
    "hiring decision", 
    "onboarding",
    "performance review",
    "remote work arrangement",
    "benefits negotiation",
    "contract terms",
    "relocation package",
    "flexible hours",
    "career development",
    "training opportunities",
    "bonus structure",
    "equity negotiation",
    "vacation time",
    "work-life balance",
    "project assignment",
    "team transition",
    "exit interview",
    "retention offer",
    "job responsibilities",
    "work equipment",
    "professional development",
    "health benefits",
    "retirement plans"
]

OUTPUT_DIR = "workplace_scenarios_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel('gemini-2.5-pro')


def build_prompt(domain: str, scenarios_needed: int) -> str:
    return f"""
Generate strictly {scenarios_needed} number of detailed, UNIQUE Workplace negotiation scenarios for the domain: "{domain}"

# UNIQUENESS RULES
1. Each scenario must differ in context, characters, and challenges.
2. Vary company sizes, industries, departments, and job roles.
3. Use distinct names, organizations, and negotiation styles.
4. Include realistic workplace contexts.
5. Must define employer & candidate roles explicitly.
6. Clearly describe what’s being negotiated.
7. Keep descriptions short (2-4 sentences).
8. Vary complexity among scenarios.

# OUTPUT FORMAT (STRICT JSON)
[
  {{
    "domain": "{domain}",
    "background": "Unique scenario description",
    "employer": "Employer's role/title",
    "candidate": "Candidate's role/title",
    "negotiation_goal": "Employer’s goal in negotiation",
    "current_position": "Candidate’s current stance"
  }},
  ...
]

Generate EXACTLY {scenarios_needed} scenarios and return ONLY valid JSON (no markdown, no extra text).
"""


def clean_json_response(response: str) -> str:
    """Clean response for JSON parsing"""
    cleaned = response.strip()
    
    
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]
        
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
        
    
    start_idx = cleaned.find('[')
    end_idx = cleaned.rfind(']') + 1
    
    if start_idx >= 0 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx]
    
    return cleaned.strip()

def extract_wait_time(error_message: str) -> int:
    """
    Extract wait time from error message like:
    "try again after 25 seconds" or "retry after 30s"
    Returns default 30 seconds if no time found
    """
    
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
            wait_time = int(match.group(1))
            print(f"📋 Extracted wait time from error: {wait_time} seconds")
            return wait_time
    
    
    print("📋 No specific wait time found in error, using default 30 seconds")
    return 30


def call_llm(prompt: str, max_retries: int = 5) -> str:
    """Make LLM call with retry logic and rate limit handling"""
    for attempt in range(max_retries):
        try:
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=8192,
                )
            )
            return response.text

        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            if any(keyword in error_str for keyword in ["429", "rate", "quota", "resource_exhausted", "retry", "try again"]):
                
                wait_time = extract_wait_time(error_msg)
                print(f"⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)  
    
    return ""



def generate_scenarios(domain: str, scenarios_needed: int = SCENARIOS_PER_DOMAIN):
    prompt = build_prompt(domain, scenarios_needed)
    
    try:
        print(f"🔄 Generating scenarios for: {domain}")
        
        raw_response = call_llm(prompt)
        if not raw_response:
            print(f"❌ Empty response for {domain}")
            return

        print(f"📥 Raw response received, cleaning JSON...")
        cleaned_text = clean_json_response(raw_response)
        print("DEBUG: ",cleaned_text)
        try:
            scenarios = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse failed: {e}")
            print(f"Cleaned text preview: {cleaned_text[:200]}...")
            return

        if len(scenarios) != scenarios_needed:
            print(f"⚠️ Expected {scenarios_needed}, got {len(scenarios)} scenarios")


        filename = f"{domain.replace(' ', '_').lower()}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(scenarios)} scenarios for: {domain}")

    except Exception as e:
        print(f"❌ Final error for {domain}: {e}")


def main():
    if not DOMAINS:
        print("❌ Domain list is empty!")
        return

    print(f"🚀 Starting generation for {len(DOMAINS)} domains...")
    
    successful = 0
    failed = 0
    
    for i, domain in enumerate(DOMAINS):
        print(f"\n{'='*50}")
        print(f"📋 Processing {i+1}/{len(DOMAINS)}: {domain}")
        print(f"{'='*50}")
        
        try:
            generate_scenarios(domain)
            successful += 1
        except Exception as e:
            print(f"❌ Critical failure for {domain}: {e}")
            failed += 1
        

        if i < len(DOMAINS) - 1:  
            print("⏳ Waiting 10 seconds before next domain...")
            time.sleep(10)
    
    print(f"\n🎉 Generation complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()
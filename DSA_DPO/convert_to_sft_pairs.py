#!/usr/bin/env python3
"""
Convert hr_dsa_dpo_segments.json (DPO preference format) to
the interleaved SFT pairs format required by the DSA-DPO LLaMA-Factory fork.

The DSA-DPO fork's sft_2_dialogue_dpo() expects:
  - index 2i   = REJECTED full conversation (context + rejected response)
  - index 2i+1 = CHOSEN  full conversation (context + chosen  response)
"""
import json, os

SRC = "./training/LLaMA-Factory/data/hr_dsa_dpo_segments.json"
DST = "./training/LLaMA-Factory/data/hr_dsa_dpo_sft_pairs.json"
DATASET_INFO = "./training/LLaMA-Factory/data/dataset_info.json"
DATASET_NAME = "hr_dsa_dpo_sft_pairs"

with open(SRC) as f:
    src = json.load(f)

print(f"Loaded {len(src)} DPO pairs from {SRC}")

sft_pairs = []
skipped = 0

for i, ex in enumerate(src):
    context = ex.get("conversations", [])
    chosen  = ex.get("chosen", {})
    rejected = ex.get("rejected", {})

    # Normalise chosen/rejected to {from, value} dict
    def normalise(r):
        if isinstance(r, dict):
            if "from" in r and "value" in r:
                return r
            if "role" in r:
                role_map = {"assistant": "gpt", "human": "human", "employer": "gpt", "candidate": "human"}
                return {"from": role_map.get(r["role"], r["role"]), "value": r.get("content", "")}
        return None

    c = normalise(chosen)
    r = normalise(rejected)

    if not c or not r:
        print(f"  ⚠ Skipping pair {i}: missing chosen/rejected")
        skipped += 1
        continue

    if not c["value"].strip() or not r["value"].strip():
        print(f"  ⚠ Skipping pair {i}: empty chosen/rejected value")
        skipped += 1
        continue

    # Sharegpt format requires conversations to start with "human" and alternate
    # human/gpt/human/gpt..., ending with gpt (even total count for SFT).
    # Our context often starts with "gpt" (employer speaks first).
    # Fix: prepend a neutral opening human turn if first turn is gpt.
    def fix_sharegpt_roles(conv):
        if not conv:
            return conv
        if conv[0]["from"] == "gpt":
            conv = [{"from": "human", "value": "Let's continue our HR negotiation conversation."}] + conv
        # Ensure alternating roles and even count
        # If it ends with "human", remove the last turn
        if conv and conv[-1]["from"] == "human":
            conv = conv[:-1]
        return conv

    rej_conv = fix_sharegpt_roles(context + [r])
    cho_conv = fix_sharegpt_roles(context + [c])

    # Rejected conversation (even index)
    sft_pairs.append({"conversations": rej_conv})
    # Chosen conversation (odd index)
    sft_pairs.append({"conversations": cho_conv})

print(f"Generated {len(sft_pairs)} SFT entries ({len(sft_pairs)//2} DPO pairs)")
if skipped:
    print(f"  ⚠ Skipped {skipped} pairs")

with open(DST, "w") as f:
    json.dump(sft_pairs, f, indent=2, ensure_ascii=False)
print(f"Saved to {DST}")

# Update dataset_info.json
with open(DATASET_INFO) as f:
    info = json.load(f)

info[DATASET_NAME] = {
    "file_name": os.path.basename(DST),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}

with open(DATASET_INFO, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"Registered '{DATASET_NAME}' in dataset_info.json")

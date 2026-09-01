#!/usr/bin/env python
"""Token-diff capture for the KVARN_ROTATION A/B.

Captures identical greedy generations against the RUNNING server and writes
one JSONL line per prompt. Run once per rotation (fresh server each time):

    KVARN_ROTATION=hadamard ...  python bench/rotor_diff_capture.py rq-had   # or rely on the default
    KVARN_ROTATION=planar   ...  python bench/rotor_diff_capture.py rq-plan
    KVARN_ROTATION=iso      ...  python bench/rotor_diff_capture.py rq-iso

Then diff: identical/healthy rotations show only occasional single-token
divergence (different quantization → different argmax somewhere is normal);
a CORRUPTED rotation drifts into incoherent text (repetition loops,
gibberish) and stays there — that drift pattern is the bug signature, and
it is what the iso 67% GSM8K run would produce if the rotation math, not
the run, is at fault.

No GPU required on this box; the server (on the GPU box) does everything.
"""

import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")


def post(path, body, timeout=300):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {KEY}"} if KEY else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Fixed prompt set — mixed task types, deterministic, same order every run:
# arithmetic, multi-step math, logical reasoning, long-form prose, counting,
# word play. 20 prompts x 200 tokens = a few minutes per rotation.
PROMPTS = [
    "What is 17 * 23 + 45? Show your work.",
    "A train leaves at 9:15 and travels for 2h 40m. What time does it arrive?",
    "Solve: 3x - 7 = 2x + 11. What is x?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes for 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "List the first 15 prime numbers.",
    "What is 2 to the power of 10, and 10 to the power of 3? Which is larger?",
    "Divide 100 by 7 and give the result to 4 decimal places.",
    "A rectangle has area 48 and perimeter 28. What are its side lengths?",
    "How many days between Jan 1 and Mar 1 of a non-leap year?",
    "Sum of all integers from 1 to 100.",
    "A number is doubled, 5 is added, the result is tripled and equals 51. What is the number?",
    "Explain in three short paragraphs why the sky is blue.",
    "Write a four-line poem about a lighthouse in winter.",
    "What is the square root of 2 to 6 decimal places?",
    "How many seconds are in 3.5 hours?",
    "If A is 3 years older than B, and in 5 years A will be twice B's age, how old are they now?",
    "Spell backwards: supercalifragilistic.",
    "A coin is flipped 10 times. What is the probability of exactly 5 heads? Give a fraction and a percentage.",
    "Describe, in one sentence each, the following: a nebula, a sonnet, a piston, a theorem.",
]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "rq"
    out_path = os.path.join(HERE, "quality-data", f"rotordiff_{tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # model name: let the server's own config decide (single-model deployments)
    model = "qwen3.8-27b"
    with open(out_path, "w") as out:
        for i, prompt in enumerate(PROMPTS):
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0,
                # thinking OFF: deterministic, short, comparable across runs
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.time()
            r = post("/completions", body)
            text = r["choices"][0]["text"]
            n = r.get("usage", {}).get("completion_tokens", len(text))
            line = {
                "i": i,
                "prompt": prompt,
                "text": text,
                "n_tokens": n,
                "wall_s": round(time.time() - t0, 2),
            }
            out.write(json.dumps(line) + "\n")
            out.flush()
            print(f"[{i + 1}/{len(PROMPTS)}] {n} tok in {line['wall_s']}s :: {text[:60]!r}...")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

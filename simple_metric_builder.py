import re

def metric_format_strict(tc: dict) -> float:
    text = tc.get("generated_output", "").lower()
    pattern = r"soal:\s+.+\s+jawaban:\s+.+\s+alur_berpikir:\s+.+"
    return 1.0 if re.search(pattern, text, re.DOTALL) else 0.0

def metric_format_compliance(tc: dict) -> float:
    text = tc.get("actual_output", "")

    required = ["soal:", "jawaban:", "alur_berpikir:"]
    found = sum(1 for r in required if r in text.lower())

    return found / len(required)  # 0.0 → 1.0

def metric_latency(tc: dict) -> float:
    t = tc.get("generation_time_seconds", None)
    if t is None:
        return None

    # normalize: lower is better
    # cap at 10s for sanity
    return min(t / 10.0, 1.0)

def make_judge_metric(client, judge_model: str):
    """
    Returns a callable metric function:
        fn(test_case: dict) -> float (0.0–1.0)

    Uses LLM-as-judge to evaluate:
    - context relevance
    - answer correctness
    - clarity for dyslexic learners
    """

    def judge_fn(tc: dict) -> float | None:
        context = tc.get("context", "")
        prompt = tc.get("prompt", "")
        output = tc.get("actual_output", "")

        judge_prompt = f"""
You are an evaluator.

Evaluate the quality of this output based on:
1. Relevance to the context
2. Correctness of the answer
3. Clarity for children with dyslexia (simple, readable)

CONTEXT:
{context}

PROMPT:
{prompt}

OUTPUT:
{output}

Return ONLY a number between 0 and 1.
Do not explain.
"""

        try:
            resp = client.chat(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            raw = resp["message"]["content"].strip()

            # brutal parsing, because LLMs love being annoying
            score = float(raw)

            # clamp to [0,1]
            return max(0.0, min(score, 1.0))

        except Exception as e:
            print(f"[WARN] Judge metric failed on {tc.get('id', '?')}: {e}")
            return None

    return judge_fn
    
# extra_metrics = [
#     {
#         "name": "format_compliance",
#         "weight": 1.0,
#         "inverted": False,
#         "threshold": 1.0,
#         "fn": metric_format_compliance,
#     },
#     {
#         "name": "format_strict",
#         "weight": 1.5,
#         "inverted": False,
#         "threshold": 1.0,
#         "fn": metric_format_strict,
#     },
#     {
#         "name": "latency",
#         "weight": 0.5,
#         "inverted": True,
#         "threshold": 0.5,
#         "fn": metric_latency,
#     },
# ]
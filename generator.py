from config import SYSTEM_PROMPT_DYSLEXIA
import time
from ollama import Client

def call_llm_ollama(ollama_client, model, prompt):
    """Generate output from an Ollama-hosted model."""
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DYSLEXIA},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7, "num_predict": 300},
        )
        # Support both dict-style and attribute-style response objects
        if isinstance(response, dict):
            return response["message"]["content"].strip()
        return response.message.content.strip()
    except Exception as e:
        return f"[GENERATION ERROR: {str(e)}]"


def call_llm(client, model, prompt):
    """Generate output from an OpenAI-compatible model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DYSLEXIA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def generate_output(llm_client, model, prompt, use_ollama=False):
    """Dispatch to the appropriate backend."""
    if use_ollama:
        return call_llm_ollama(llm_client, model, prompt)
    return call_llm(llm_client, model, prompt)


def build_test_cases(client, model, templates):
    test_cases = []

    # ── user simulation (prompts array) ──────────────
    for template in templates:
        for i, prompt in enumerate(template["prompts"], 1):
            tc_id = f"TC_{template['id_prefix']}_USER_{model}_{i:03d}"
            actual_output = call_llm_ollama(client, model, prompt)
            
            test_cases.append({
                "id": tc_id,
                "model":model,
                "category": template["category"],
                "sub_category": template["sub_category"],
                "difficulty_level": template["difficulty_level"],
                "track": "user_simulation",       
                "prompt_strategy": "natural",
                "input": prompt,
                "actual_output": actual_output,
                "expected_output": template["expected_output"],
                "context": template["context"],
                "retrieval_context": template["context"],
            })

    # ── structured_prompts ──────────
    for template in templates:
        sp = template.get("structured_prompts", {})
        for strategy, prompt in sp.items():
            if not prompt:
                continue
            tc_id = f"TC_{template['id_prefix']}_{strategy.upper()}_{model}"
            actual_output = call_llm_ollama(client, model, prompt)
            time.sleep(0.2)
            test_cases.append({
                "id": tc_id,
                "model":model,
                "category": template["category"],
                "sub_category": template["sub_category"],
                "difficulty_level": template["difficulty_level"],
                "track": "research",        
                "prompt_strategy": strategy,       # zero_shot / few_shot / cot / hybrid
                "input": prompt,
                "actual_output": actual_output,
                "expected_output": template["expected_output"],
                "context": template["context"],
                "retrieval_context": template["context"],
                
            })

    return test_cases


# ── Convenience alias kept for backward compatibility ─────────────────────────
def generate_testcases(client, model, prompts, use_ollama=False):
    results = []
    for i, prompt in enumerate(prompts):
        output = generate_output(client, model, prompt, use_ollama)
        results.append({"id": f"GEN_{i}", "input": prompt, "output": output})
    return results

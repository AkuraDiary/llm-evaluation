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


def build_test_cases(client, model, templates, use_ollama=False):
    """
    Generate LLM outputs for every prompt in every template and return
    a list of fully-populated test-case dicts ready for evaluation.
    """
    test_cases = []
    case_counter = {}

    for template in templates:
        prefix = template["id_prefix"]
        case_counter[prefix] = 0

        for prompt in template["prompts"]:
            case_counter[prefix] += 1
            tc_id = (
                f"TC_{prefix}_"
                f"{model.replace(':', '_').replace('-', '_').upper()}_"
                f"{case_counter[prefix]:03d}"
            )

            if use_ollama:
                actual_output = call_llm_ollama(client, model, prompt)
            else:
                actual_output = call_llm(client, model, prompt)

            # Brief pause to avoid overwhelming local Ollama server
            time.sleep(0.2)

            test_cases.append({
                "id": tc_id,
                "category": template["category"],
                "sub_category": template["sub_category"],
                "difficulty_level": template["difficulty_level"],
                "model": model,
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

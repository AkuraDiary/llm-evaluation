# test_generation.py

from generator import generate_testcases
from openai import OpenAI
from ollama import Client
from config import TEMPLATE_TEST_CASES
from util import json_exporter
from config import OLLAMA_API_KEY, OLLAMA_HOST, MODELS_TO_EVALUATE_OLLAMA
from generator import build_test_cases


def main():
    ollama_client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )
    model = MODELS_TO_EVALUATE_OLLAMA[0]
    test_cases = build_test_cases(client=ollama_client, model=model, templates=TEMPLATE_TEST_CASES[1:2], use_ollama=True)
    
    print(f"[INFO] Total test cases untuk {model}: {len(test_cases)}")

    print(f"\n[INFO] Total seluruh test cases: {len(test_cases)}")
    
    json_exporter(test_cases, "testcases_generated.json")


if __name__ == "__main__":
    main()
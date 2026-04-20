# test_generation.py

from generator import generate_testcases
from openai import OpenAI
from ollama import Client
from config import TEMPLATE_TEST_CASES
from util import json_exporter
from config import OPENAI_API_KEY, OLLAMA_API_KEY, OLLAMA_HOST
from generator import build_test_cases

TEST_PROMPTS = [
    "Buat soal penjumlahan apel",
    "Buat soal pengurangan bola",
]


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    ollama_client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )
    model = "gemma4:31b-cloud"
    test_cases = build_test_cases(client=ollama_client, model=model, templates=TEMPLATE_TEST_CASES[1:2], use_ollama=True)
    
    print(f"[INFO] Total test cases untuk {model}: {len(test_cases)}")

    print(f"\n[INFO] Total seluruh test cases: {len(test_cases)}")
    
    json_exporter(test_cases, "testcases_generated.json")


if __name__ == "__main__":
    main()
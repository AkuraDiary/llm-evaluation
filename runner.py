# runner.py
from config import MODELS_TO_EVALUATE,MODELS_TO_EVALUATE_OLLAMA, TEMPLATE_TEST_CASES, METRICS_CONFIG
from generator import build_test_cases
from evaluator_core import evaluate_test_cases
from metrics_factory import create_metrics, create_weights_map
from aggregators import group_by_model
from aggregators import group_by_metric
from openai import OpenAI
from simple_report import build_simple_report
from academic_report import build_academic_report
from ollama import Client
import report_builder
import json
from util import json_exporter
from config import OPENAI_API_KEY, OLLAMA_API_KEY, OLLAMA_HOST

client = OpenAI(api_key=OPENAI_API_KEY)

ollama_client = Client(
    host=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

def create_metrics():
    return [
        {
            "instance": cfg["class"](threshold=cfg["threshold"]),
            "threshold": cfg["threshold"],
            "weight": cfg["weight"],
        }
        for cfg in METRICS_CONFIG
    ]

def run_generation_only():
    client = ollama_client
    model = MODELS_TO_EVALUATE_OLLAMA[0]

    test_cases = build_test_cases(
        client,
        model,
        TEMPLATE_TEST_CASES[1:2],
        use_ollama=True
    )

    return test_cases

USE_ACADEMIC_REPORT = False

def run():
    client = ollama_client

    metrics = create_metrics()
    weights = create_weights_map()

    all_cases = []

    for model in MODELS_TO_EVALUATE_OLLAMA:
        cases = build_test_cases(client, model, TEMPLATE_TEST_CASES[1:2], use_ollama=True)
        all_cases.extend(cases)
    
    # (
    #     all_test_case_results,
    #     raw_scores_by_metric,
    #     scores_by_category,
    #     scores_by_difficulty,
    #     scores_by_model,
    # ) = evaluate_test_cases(all_cases, weights, metrics)
    
    # aggregate_statistics = compute_aggregate_statistics(raw_scores_by_metric)
    # metric_correlation_matrix = compute_correlation_matrix(raw_scores_by_metric)
    # model_comparison = compute_model_comparison(scores_by_model)
    # performance_by_category = compute_category_performance(scores_by_category)
    # performance_by_difficulty = compute_difficulty_performance(scores_by_difficulty)
    json_exporter(all_cases, "all_cases.json")
    results = evaluate_test_cases(all_cases, weights, metrics)

    # grouped = group_by_metric(results)

    # report = ""
    # if USE_ACADEMIC_REPORT:
    #     # print("working on it")
    #     report = build_academic_report(
    #         all_test_case_results,
            
    #     )
    # else:
        # report = build_simple_report(results, grouped)

    json_exporter(results, "all_test_case_result.json")



if __name__ == "__main__":
    print(run_generation_only())
import json
import sys
import time
import platform
from datetime import datetime, timezone

from ollama import Client as OllamaClient

from config import (
    OLLAMA_HOST,
    OLLAMA_API_KEY,
    MODELS_TO_EVALUATE_OLLAMA,
    TEMPLATE_TEST_CASES,
    METRIC_NAMES,
    METRIC_WEIGHTS,
    INVERTED_METRICS,
    EXPERIMENT_START_UTC,
    SYSTEM_PROMPT_DYSLEXIA,
    JUDGE_MODEL,
)
from generator import build_test_cases
from evaluator_core import (
    evaluate_all,
    compute_descriptive,
    confidence_interval_95,
    bootstrap_ci,
)
from statistics import safe_mean
from academic_report import (
    build_aggregate_statistics,
    build_correlation_matrix,
    build_model_comparison,
    build_group_analysis,
)
from util import json_exporter


def _make_ollama_client():
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return OllamaClient(host=OLLAMA_HOST, headers=headers)

def run():
    t0 = time.time()

    ollama_client = _make_ollama_client()

    # ── 2. Generate test cases ────────────────────────────────────────────────
    all_test_cases = []
    for model in MODELS_TO_EVALUATE_OLLAMA:
        print(f"\n[INFO] Generating test cases untuk model: {model}")
        tcs = build_test_cases(ollama_client, model, TEMPLATE_TEST_CASES[0:1], use_ollama=True)
        all_test_cases.extend(tcs)
        print(f"[INFO] Total test cases untuk {model}: {len(tcs)}")

    print(f"\n[INFO] Total seluruh test cases: {len(all_test_cases)}")
    print("[INFO] Mulai evaluasi komprehensif...\n")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    all_results, raw_by_metric, by_category, by_difficulty, by_model = evaluate_all(
        all_test_cases, ollama_client
    )

    # ── 4. Aggregate statistics ───────────────────────────────────────────────
    print("\n[INFO] Menghitung statistik agregat...")
    aggregate_stats = build_aggregate_statistics(raw_by_metric)
    correlation_matrix = build_correlation_matrix(raw_by_metric)
    model_comparison, global_bonferroni = build_model_comparison(by_model, raw_by_metric)
    category_analysis = build_group_analysis(by_category, "category")
    difficulty_analysis = build_group_analysis(by_difficulty, "difficulty")

    # ── 5. Summary figures ────────────────────────────────────────────────────
    all_composite = [
        tc["test_case_summary"]["composite_weighted_score"]
        for tc in all_results
        if tc["test_case_summary"]["composite_weighted_score"] is not None
    ]
    all_pass_rates = [tc["test_case_summary"]["pass_rate"] for tc in all_results]
    all_times = [tc["test_case_summary"]["execution_time_seconds"] for tc in all_results]
    total_passed = sum(tc["test_case_summary"]["metrics_passed"] for tc in all_results)
    total_evals = sum(tc["test_case_summary"]["total_metrics"] for tc in all_results)
    total_time = round(time.time() - t0, 4)

    comp_ci = confidence_interval_95(all_composite)
    comp_bci = bootstrap_ci(all_composite)
    comp_desc = compute_descriptive(all_composite)

    raw_matrix = {
        tc["test_case_id"]: {m["metric_name"]: m["raw_score"] for m in tc["metrics_detail"]}
        for tc in all_results
    }

    # ── 6. Assemble output document ──────────────────────────────────────────
    output_doc = {
        "experiment_metadata": {
            "experiment_id": "EXP_DYSLEXIA_LLM_EVAL_OLLAMA_V4_Q1",
            "title": (
                "Comprehensive Free Multi-Model Evaluation of LLM-Generated Educational "
                "Content for Children with Dyslexia Using Local Ollama Models"
            ),
            "description": (
                "A fully open-source, zero-cost systematic multi-model evaluation of Large Language Model "
                "outputs for generating educationally appropriate questions and answers tailored to "
                "children with dyslexia. All models run locally via Ollama (no API cost). "
                "Evaluation employs 14 metrics: NLP similarity (ROUGE-1/2/L, BLEU), readability "
                "(Flesch RE, FK Grade, Gunning Fog, SMOG, ARI, Coleman-Liau), dyslexia compliance, "
                "LLM-as-Judge (local), toxicity, bias, context relevancy, faithfulness, answer relevancy, "
                "and lexical diversity. "
                "Statistics: descriptive, Shapiro-Wilk, Mann-Whitney U, Kruskal-Wallis H, Cohen's d, "
                "Pearson r, Spearman rho, p-values, Bonferroni, 95% CI (t), Bootstrap CI (2000)."
            ),
            "research_domain": "Special Education Technology / NLP / Computational Linguistics",
            "task_type": "Educational Question-Answer Generation for Dyslexic Learners",
            "target_population": "Children with Dyslexia, Elementary School Level, Grade 1-3, Indonesia",
            "evaluation_infrastructure": "Ollama (Local, Free, Open-Source)",
            "models_evaluated": MODELS_TO_EVALUATE_OLLAMA,
            "judge_model": JUDGE_MODEL,
            "generator_system_prompt": SYSTEM_PROMPT_DYSLEXIA,
            "timestamp_start_utc": EXPERIMENT_START_UTC,
            "timestamp_end_utc": datetime.now(timezone.utc).isoformat(),
            "total_execution_time_seconds": total_time,
            "python_version": sys.version,
            "platform": platform.platform(),
            "total_test_cases": len(all_results),
            "total_metrics_per_case": len(METRIC_NAMES),
            "total_evaluations_performed": total_evals,
            "cost_usd": 0.0,
            "reproducibility_note": "All models are local. Pin Ollama model versions for full reproducibility.",
        },
        "evaluation_configuration": {
            "metrics_applied": [
                {
                    "metric_name": m,
                    "weight": METRIC_WEIGHTS.get(m, 1.0),
                    "is_inverted": m in INVERTED_METRICS,
                    "threshold": 0.1 if m in {"toxicity", "bias"} else 0.5,
                }
                for m in METRIC_NAMES
            ],
            "composite_score_formula": (
                "Weighted average of normalized metric scores; "
                "inverted metrics use (1-score) before normalization"
            ),
            "statistical_methods": [
                "Descriptive statistics: mean, std, median, min, max, range, Q1, Q3, IQR, CV, skewness, kurtosis",
                "Normality: Shapiro-Wilk approximation",
                "Non-parametric pairwise: Mann-Whitney U + z-approx p-value",
                "Non-parametric multi-group: Kruskal-Wallis H + chi-square approx p-value",
                "Effect size: Cohen's d (pooled std)",
                "Parametric correlation: Pearson r + t-test p-value",
                "Non-parametric correlation: Spearman rho + t-approx p-value",
                "Multiple comparisons: Bonferroni correction",
                "95% CI: t-distribution (exact t-critical for n<30)",
                "Robust CI: Bootstrap 95% (2000 resamples, percentile, seed=42)",
            ],
            "dyslexia_compliance_rubric": {
                "avg_sentence_length_le6_words": "30 points",
                "flesch_reading_ease_ge80": "25 points",
                "has_explicit_answer": "20 points",
                "numbers_in_range_1_20": "15 points",
                "no_forbidden_complex_words": "10 points",
            },
        },
        "test_cases": all_results,
        "aggregate_statistics_per_metric": aggregate_stats,
        "metric_correlation_matrix": correlation_matrix,
        "model_comparison_analysis": model_comparison,
        "global_bonferroni_correction_all_pairwise": global_bonferroni,
        "performance_by_category": category_analysis,
        "performance_by_difficulty_level": difficulty_analysis,
        "overall_experiment_summary": {
            "total_test_cases": len(all_results),
            "models_evaluated": MODELS_TO_EVALUATE_OLLAMA,
            "total_evaluations_performed": total_evals,
            "total_metrics_passed": total_passed,
            "total_metrics_failed": total_evals - total_passed,
            "overall_pass_rate": round(total_passed / total_evals, 6) if total_evals > 0 else 0,
            "composite_score_statistics": comp_desc,
            "composite_score_ci_95_t": {"lower": comp_ci[0], "upper": comp_ci[1]},
            "composite_score_bootstrap_ci_95": {"lower": comp_bci[0], "upper": comp_bci[1]},
            "pass_rate_statistics": compute_descriptive(all_pass_rates),
            "execution_time_statistics": {
                "total_seconds": total_time,
                "mean_per_case_seconds": round(safe_mean(all_times), 4) if all_times else None,
                "min_seconds": round(min(all_times), 4) if all_times else None,
                "max_seconds": round(max(all_times), 4) if all_times else None,
            },
        },
        "raw_scores_matrix": raw_matrix,
        "readability_summary_per_model": {
            mdl: {
                field: compute_descriptive([
                    tc["readability_analysis"][field]
                    for tc in all_results if tc["model"] == mdl
                ])
                for field in [
                    "flesch_reading_ease",
                    "avg_sentence_length_words",
                    "flesch_kincaid_grade",
                    "gunning_fog_index",
                    "smog_index",
                    "automated_readability_index",
                    "coleman_liau_index",
                    "type_token_ratio",
                    "sentences_exceeding_8_words_pct",
                ]
            }
            for mdl in MODELS_TO_EVALUATE_OLLAMA
        },
    }

    # ── 7. Export ─────────────────────────────────────────────────────────────
    output_path = "evaluasi_ollama_llm_disleksia.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"  Evaluasi selesai!")
    print(f"  Output: {output_path}")
    print(f"  Total test cases  : {len(all_results)}")
    print(f"  Model dievaluasi  : {MODELS_TO_EVALUATE_OLLAMA}")
    print(f"  Total evaluasi    : {total_evals}")
    print(f"  Total evaluasi    : {total_evals}")
    print(f"  Overall pass rate : {round(total_passed / total_evals * 100, 2) if total_evals else 0}%")
    print(f"  Composite mean    : {round(safe_mean(all_composite), 4) if all_composite else 'N/A'}")
    print(f"  Total waktu       : {total_time} detik")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run()

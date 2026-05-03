"""
run.py — Dyslexia LLM Evaluation Pipeline
==========================================
Three independently runnable steps:

  python run.py generate          # Step 1: generate + save test-case outputs
  python run.py evaluate          # Step 2: run judge LLM evaluation
  python run.py metrics           # Step 3: compute plug-and-play metrics + final report
  python run.py all               # Run all three steps end-to-end

Each step reads from the previous step's JSON output, so you can:
  - Re-run evaluation without re-generating (expensive) LLM outputs
  - Swap in/out metrics without re-running the judge
  - Inspect intermediate outputs at any stage

Output files
------------
  outputs/step1_generated.json    raw model outputs + test case metadata
  outputs/step2_evaluated.json    step1 + judge LLM scores appended
  outputs/step3_final.json        step2 + all metrics + aggregate stats

Backward compatibility
----------------------
  The final output document structure is identical to the original
  evaluasi_ollama_llm_disleksia.json schema. All downstream consumers
  (academic_report, json_exporter) work unchanged.
"""

import argparse
import json
import os
import platform
import sys
from report_builder import simple_output
import time
from datetime import datetime, timezone
from config import JUDGE_MODEL
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

from simple_metric_builder import (
    metric_format_compliance,
    metric_latency,
    metric_format_strict,
    make_judge_metric
)

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = "outputs"
STEP1_PATH = os.path.join(OUT_DIR, "step1_generated.json")
STEP2_PATH = os.path.join(OUT_DIR, "step2_evaluated.json")
STEP3_PATH = os.path.join(OUT_DIR, "step3_final.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ollama_client() -> OllamaClient:
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return OllamaClient(host=OLLAMA_HOST, headers=headers)


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Required input file not found: {path}\n"
            f"        Run the previous step first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, doc: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {path}")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate
# ─────────────────────────────────────────────────────────────────────────────

def step_generate(
    *,
    from_json: str | None = None,
    template_slice: slice = slice(None),
) -> str:
    """
    Generate model outputs for every test case and save to step1_generated.json.

    Parameters
    ----------
    from_json : str or None
        Path to a previously saved step1_generated.json to load instead of
        re-generating. Useful when you already have outputs and just want to
        re-package them.
    template_slice : slice
        Which templates to use, e.g. slice(0, 1) for the first template only.
        Defaults to all templates.

    Returns the output path.
    """
    _ensure_out_dir()
    t0 = time.time()

    if from_json:
        # ── Load existing outputs instead of re-generating ──────────────────
        print(f"[INFO] Loading pre-generated test cases from: {from_json}")
        doc = _load_json(from_json)
        _save_json(STEP1_PATH, doc)
        return STEP1_PATH

    # ── Generate fresh ───────────────────────────────────────────────────────
    client = _make_ollama_client()
    templates = TEMPLATE_TEST_CASES[template_slice]

    all_test_cases = []
    for model in MODELS_TO_EVALUATE_OLLAMA:
        print(f"\n[STEP 1] Generating test cases — model: {model}")
        tcs = build_test_cases(client, model, templates)
        all_test_cases.extend(tcs)
        print(f"[STEP 1] Generated {len(tcs)} test cases for {model}")

    print(f"\n[STEP 1] Total test cases: {len(all_test_cases)}")

    doc = {
        "step": "generate",
        "timestamp_utc": _timestamp(),
        "models_evaluated": MODELS_TO_EVALUATE_OLLAMA,
        "generator_system_prompt": SYSTEM_PROMPT_DYSLEXIA,
        "python_version": sys.version,
        "platform": platform.platform(),
        "generation_time_seconds": round(time.time() - t0, 4),
        "total_test_cases": len(all_test_cases),
        "test_cases": all_test_cases,
    }

    _save_json(STEP1_PATH, doc)
    print(f"[STEP 1] Done in {doc['generation_time_seconds']}s")
    return STEP1_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Evaluate (judge LLM)
# ─────────────────────────────────────────────────────────────────────────────

def step_evaluate(*, from_json: str | None = None) -> str:
    """
    Run the judge LLM evaluation over the generated test cases.
    Reads step1_generated.json (or from_json if provided).
    Saves result to step2_evaluated.json.

    Returns the output path.
    """
    _ensure_out_dir()
    t0 = time.time()

    src_path = from_json or STEP1_PATH
    print(f"[STEP 2] Loading generated test cases from: {src_path}")
    step1 = _load_json(src_path)
    test_cases = step1["test_cases"]
    print(f"[STEP 2] Loaded {len(test_cases)} test cases")

    client = _make_ollama_client()

    print(f"[STEP 2] Running judge evaluation (model: {JUDGE_MODEL})...")
    all_results, raw_by_metric, by_category, by_difficulty, by_model = evaluate_all_custom(
        test_cases, client
    )

    elapsed = round(time.time() - t0, 4)

    doc = {
        "step": "evaluate",
        "timestamp_utc": _timestamp(),
        "judge_model": JUDGE_MODEL,
        "evaluation_time_seconds": elapsed,
        # Carry forward step 1 metadata
        "models_evaluated": step1["models_evaluated"],
        "generator_system_prompt": step1["generator_system_prompt"],
        "total_test_cases": len(all_results),
        "test_cases": all_results,
        # Persist intermediate aggregations for step 3
        "_internal": {
            "raw_by_metric": raw_by_metric,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "by_model": by_model,
        },
    }

    _save_json(STEP2_PATH, doc)
    print(f"[STEP 2] Done in {elapsed}s")
    return STEP2_PATH

def evaluate_all_custom(test_cases, client):
    metrics = get_custom_metrics(client)

    all_results = []
    raw_by_metric = {}
    by_category = {}
    by_difficulty = {}
    by_model = {}

    for tc in test_cases:
        tc_result = dict(tc)
        tc_result["metrics_detail"] = []

        for m in metrics:
            name = m["name"]
            fn = m["fn"]

            try:
                score = fn(tc)
            except Exception:
                score = None

            passed = (
                score is not None and
                (
                    (score <= m["threshold"]) if m["inverted"]
                    else (score >= m["threshold"])
                )
            )

            tc_result["metrics_detail"].append({
                "metric_name": name,
                "raw_score": score,
                "is_inverted": m["inverted"],
                "threshold": m["threshold"],
                "passed": passed,
            })

            if score is not None:
                raw_by_metric.setdefault(name, []).append(score)

        # Optional grouping (keep if you need Step 3 stats)
        cat = tc.get("category")
        diff = tc.get("difficulty_level")
        model = tc.get("model")

        if cat:
            by_category.setdefault(cat, []).append(tc_result)
        if diff:
            by_difficulty.setdefault(diff, []).append(tc_result)
        if model:
            by_model.setdefault(model, []).append(tc_result)

        all_results.append(tc_result)

    return all_results, raw_by_metric, by_category, by_difficulty, by_model

def get_custom_metrics(client):
    return [
        {
            "name": "format_compliance",
            "fn": metric_format_compliance,
            "inverted": False,
            "threshold": 1.0,
        },
        {
            "name": "format_strict",
            "fn": metric_format_strict,
            "inverted": False,
            "threshold": 1.0,
        },
        # {
        #     "name": "latency",
        #     "fn": metric_latency,
        #     "inverted": True,
        #     "threshold": 0.5,
        # },
        {
            "name": "judge_overall",
            "fn": make_judge_metric(client, JUDGE_MODEL),
            "inverted": False,
            "threshold": 0.6,
        },
    ]
# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Metrics (plug-and-play)
# ─────────────────────────────────────────────────────────────────────────────

def step_metrics(
    *,
    from_json: str | None = None,
    extra_metrics: list[dict] | None = None,
    disable_metrics: list[str] | None = None,
) -> str:
    """
    Compute final metric stack and produce the full output report.
    Reads step2_evaluated.json (or from_json if provided).
    Saves result to step3_final.json.

    Plug-and-play interface
    -----------------------
    extra_metrics : list of dicts, each with keys:
        {
            "name"    : str,          # metric identifier
            "weight"  : float,        # contribution to composite score
            "inverted": bool,         # True if lower raw score = better
            "threshold": float,       # pass/fail cutoff
            "fn"      : callable,     # fn(test_case: dict) -> float  (0.0–1.0)
        }
        These are merged on top of the base METRIC_NAMES / METRIC_WEIGHTS.
        Backward compatible: adding extras never removes existing metrics.

    disable_metrics : list of str
        Metric names to exclude from composite scoring and pass/fail counts.
        The raw scores are still stored in output for traceability.

    Returns the output path.
    """
    _ensure_out_dir()
    t0 = time.time()

    src_path = from_json or STEP2_PATH
    print(f"[STEP 3] Loading evaluated results from: {src_path}")
    step2 = _load_json(src_path)

    all_results = step2["test_cases"]
    internal = step2.get("_internal", {})
    raw_by_metric = internal.get("raw_by_metric", {})
    by_category = internal.get("by_category", {})
    by_difficulty = internal.get("by_difficulty", {})
    by_model = internal.get("by_model", {})

    # ── Build active metric registry ─────────────────────────────────────────
    # Start from the base config
    active_weights = dict(METRIC_WEIGHTS)
    active_inverted = set(INVERTED_METRICS)
    active_thresholds = {
        m: (0.1 if m in {"toxicity", "bias"} else 0.5)
        for m in METRIC_NAMES
    }
    active_extra_fns: dict[str, callable] = {}

    # Apply extra metrics (additive, never breaks existing)
    if extra_metrics:
        for em in extra_metrics:
            name = em["name"]
            active_weights[name] = em.get("weight", 1.0)
            active_thresholds[name] = em.get("threshold", 0.5)
            if em.get("inverted", False):
                active_inverted.add(name)
            if "fn" in em:
                active_extra_fns[name] = em["fn"]
        print(f"[STEP 3] Extra metrics registered: {[em['name'] for em in extra_metrics]}")

    # Apply disabled metrics (excluded from scoring, raw score still stored)
    disabled = set(disable_metrics or [])
    if disabled:
        print(f"[STEP 3] Metrics disabled from scoring: {disabled}")

    # ── Run extra metric functions against each test case ────────────────────
    if active_extra_fns:
        print(f"[STEP 3] Computing {len(active_extra_fns)} extra metric(s)...")
        for tc in all_results:
            for mname, fn in active_extra_fns.items():
                try:
                    score = float(fn(tc))
                except Exception as e:
                    print(f"[WARN] Extra metric '{mname}' failed on {tc.get('id', '?')}: {e}")
                    score = None

                # Append to metrics_detail (same schema as existing metrics)
                tc.setdefault("metrics_detail", []).append({
                    "metric_name": mname,
                    "raw_score": score,
                    "weight": active_weights[mname],
                    "is_inverted": mname in active_inverted,
                    "threshold": active_thresholds[mname],
                    "passed": (
                        score is not None and
                        (
                            (score <= active_thresholds[mname]) if mname in active_inverted
                            else (score >= active_thresholds[mname])
                        )
                    ),
                    "source": "extra_metric",
                })

                # Update raw_by_metric for aggregate stats
                raw_by_metric.setdefault(mname, [])
                if score is not None:
                    raw_by_metric[mname].append(score)

    # ── Recompute composite scores with active weights ────────────────────────
    # (respects disabled metrics and new extra metric weights)
    print("[STEP 3] Recomputing composite scores...")
    for tc in all_results:
        details = tc.get("metrics_detail", [])
        weighted_sum = 0.0
        weight_total = 0.0
        passed_count = 0
        total_count = 0

        for m in details:
            mname = m["metric_name"]
            if mname in disabled:
                continue
            raw = m.get("raw_score")
            if raw is None:
                continue

            w = active_weights.get(mname, 1.0)
            normalized = (1.0 - raw) if mname in active_inverted else raw
            weighted_sum += normalized * w
            weight_total += w
            total_count += 1

            thresh = active_thresholds.get(mname, 0.5)
            passed = (raw <= thresh) if mname in active_inverted else (raw >= thresh)
            passed_count += int(passed)

        composite = round(weighted_sum / weight_total, 6) if weight_total > 0 else None
        pass_rate = round(passed_count / total_count, 6) if total_count > 0 else 0.0

        # Update summary in-place (backward compatible field names)
        tc.setdefault("test_case_summary", {}).update({
            "composite_weighted_score": composite,
            "pass_rate": pass_rate,
            "metrics_passed": passed_count,
            "total_metrics": total_count,
            "disabled_metrics": list(disabled),
            "active_extra_metrics": list(active_extra_fns.keys()),
        })

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("[STEP 3] Building aggregate statistics...")
    aggregate_stats = build_aggregate_statistics(raw_by_metric)
    correlation_matrix = build_correlation_matrix(raw_by_metric)
    model_comparison, global_bonferroni = build_model_comparison(by_model, raw_by_metric)
    category_analysis = build_group_analysis(by_category, "category")
    difficulty_analysis = build_group_analysis(by_difficulty, "difficulty")

    # ── Summary figures ───────────────────────────────────────────────────────
    all_composite = [
        tc["test_case_summary"]["composite_weighted_score"]
        for tc in all_results
        if tc["test_case_summary"].get("composite_weighted_score") is not None
    ]
    all_pass_rates = [tc["test_case_summary"]["pass_rate"] for tc in all_results]
    all_times = [
        tc["test_case_summary"].get("execution_time_seconds", 0)
        for tc in all_results
    ]
    total_passed = sum(tc["test_case_summary"]["metrics_passed"] for tc in all_results)
    total_evals = sum(tc["test_case_summary"]["total_metrics"] for tc in all_results)
    total_time = round(time.time() - t0, 4)

    comp_ci = confidence_interval_95(all_composite)
    comp_bci = bootstrap_ci(all_composite)
    comp_desc = compute_descriptive(all_composite)

    raw_matrix = {
        tc["test_case_id"]: {
            m["metric_name"]: m["raw_score"]
            for m in tc.get("metrics_detail", [])
        }
        for tc in all_results
    }

    models = step2["models_evaluated"]

    # ── Assemble final document (same schema as original only more modular) ────────────────────
    context = {
       
    }
    # output_doc = simple_output
    # output_doc = {
    #     "experiment_metadata": {
    #         "experiment_id": "EXP_DYSLEXIA_LLM_EVAL_OLLAMA_V4_Q1",
    #         "title": (
    #             "Comprehensive Free Multi-Model Evaluation of LLM-Generated Educational "
    #             "Content for Children with Dyslexia Using Local Ollama Models"
    #         ),
    #         "description": (
    #             "A fully open-source, zero-cost systematic multi-model evaluation of Large Language Model "
    #             "outputs for generating educationally appropriate questions and answers tailored to "
    #             "children with dyslexia. All models run locally via Ollama (no API cost). "
    #             "Evaluation employs 14 metrics: NLP similarity (ROUGE-1/2/L, BLEU), readability "
    #             "(Flesch RE, FK Grade, Gunning Fog, SMOG, ARI, Coleman-Liau), dyslexia compliance, "
    #             "LLM-as-Judge (local), toxicity, bias, context relevancy, faithfulness, answer relevancy, "
    #             "and lexical diversity. "
    #             "Statistics: descriptive, Shapiro-Wilk, Mann-Whitney U, Kruskal-Wallis H, Cohen's d, "
    #             "Pearson r, Spearman rho, p-values, Bonferroni, 95% CI (t), Bootstrap CI (2000)."
    #         ),
    #         "research_domain": "Special Education Technology / NLP / Computational Linguistics",
    #         "task_type": "Educational Question-Answer Generation for Dyslexic Learners",
    #         "target_population": "Children with Dyslexia, Elementary School Level, Grade 1-3, Indonesia",
    #         "evaluation_infrastructure": "Ollama (Local, Free, Open-Source)",
    #         "models_evaluated": models,
    #         "judge_model": JUDGE_MODEL,
    #         "generator_system_prompt": SYSTEM_PROMPT_DYSLEXIA,
    #         "timestamp_start_utc": EXPERIMENT_START_UTC,
    #         "timestamp_end_utc": _timestamp(),
    #         "total_execution_time_seconds": total_time,
    #         "python_version": sys.version,
    #         "platform": platform.platform(),
    #         "total_test_cases": len(all_results),
    #         "total_metrics_per_case": len(active_weights) - len(disabled),
    #         "total_evaluations_performed": total_evals,
    #         "cost_usd": 0.0,
    #         "reproducibility_note": "All models are local. Pin Ollama model versions for full reproducibility.",
    #     },
    #     "evaluation_configuration": {
    #         "metrics_applied": [
    #             {
    #                 "metric_name": m,
    #                 "weight": active_weights.get(m, 1.0),
    #                 "is_inverted": m in active_inverted,
    #                 "threshold": active_thresholds.get(m, 0.5),
    #                 "disabled": m in disabled,
    #                 "source": "extra_metric" if m in active_extra_fns else "base",
    #             }
    #             for m in active_weights
    #         ],
    #         "composite_score_formula": (
    #             "Weighted average of normalized metric scores; "
    #             "inverted metrics use (1-score) before normalization; "
    #             "disabled metrics excluded from composite"
    #         ),
    #         "disabled_metrics": list(disabled),
    #         "extra_metrics_registered": [em["name"] for em in (extra_metrics or [])],
    #         "statistical_methods": [
    #             "Descriptive statistics: mean, std, median, min, max, range, Q1, Q3, IQR, CV, skewness, kurtosis",
    #             "Normality: Shapiro-Wilk approximation",
    #             "Non-parametric pairwise: Mann-Whitney U + z-approx p-value",
    #             "Non-parametric multi-group: Kruskal-Wallis H + chi-square approx p-value",
    #             "Effect size: Cohen's d (pooled std)",
    #             "Parametric correlation: Pearson r + t-test p-value",
    #             "Non-parametric correlation: Spearman rho + t-approx p-value",
    #             "Multiple comparisons: Bonferroni correction",
    #             "95% CI: t-distribution (exact t-critical for n<30)",
    #             "Robust CI: Bootstrap 95% (2000 resamples, percentile, seed=42)",
    #         ],
    #         "dyslexia_compliance_rubric": {
    #             "avg_sentence_length_le6_words": "30 points",
    #             "flesch_reading_ease_ge80": "25 points",
    #             "has_explicit_answer": "20 points",
    #             "numbers_in_range_1_20": "15 points",
    #             "no_forbidden_complex_words": "10 points",
    #         },
    #     },
    #     "test_cases": all_results,
    #     "aggregate_statistics_per_metric": aggregate_stats,
    #     "metric_correlation_matrix": correlation_matrix,
    #     "model_comparison_analysis": model_comparison,
    #     "global_bonferroni_correction_all_pairwise": global_bonferroni,
    #     "performance_by_category": category_analysis,
    #     "performance_by_difficulty_level": difficulty_analysis,
    #     "overall_experiment_summary": {
    #         "total_test_cases": len(all_results),
    #         "models_evaluated": models,
    #         "total_evaluations_performed": total_evals,
    #         "total_metrics_passed": total_passed,
    #         "total_metrics_failed": total_evals - total_passed,
    #         "overall_pass_rate": round(total_passed / total_evals, 6) if total_evals > 0 else 0,
    #         "composite_score_statistics": comp_desc,
    #         "composite_score_ci_95_t": {"lower": comp_ci[0], "upper": comp_ci[1]},
    #         "composite_score_bootstrap_ci_95": {"lower": comp_bci[0], "upper": comp_bci[1]},
    #         "pass_rate_statistics": compute_descriptive(all_pass_rates),
    #         "execution_time_statistics": {
    #             "total_seconds": total_time,
    #             "mean_per_case_seconds": round(safe_mean(all_times), 4) if all_times else None,
    #             "min_seconds": round(min(all_times), 4) if all_times else None,
    #             "max_seconds": round(max(all_times), 4) if all_times else None,
    #         },
    #     },
    #     "raw_scores_matrix": raw_matrix,
    #     "readability_summary_per_model": {
    #         mdl: {
    #             field: compute_descriptive([
    #                 tc["readability_analysis"][field]
    #                 for tc in all_results
    #                 if tc.get("model") == mdl and "readability_analysis" in tc
    #             ])
    #             for field in [
    #                 "flesch_reading_ease",
    #                 "avg_sentence_length_words",
    #                 "flesch_kincaid_grade",
    #                 "gunning_fog_index",
    #                 "smog_index",
    #                 "automated_readability_index",
    #                 "coleman_liau_index",
    #                 "type_token_ratio",
    #                 "sentences_exceeding_8_words_pct",
    #             ]
    #         }
    #         for mdl in models
    #     },
    # }

    _save_json(STEP3_PATH, output_doc)

    print(f"\n{'=' * 65}")
    print(f"  [STEP 3] Evaluasi selesai!")
    print(f"  Output            : {STEP3_PATH}")
    print(f"  Total test cases  : {len(all_results)}")
    print(f"  Model dievaluasi  : {models}")
    print(f"  Total evaluasi    : {total_evals}")
    print(f"  Overall pass rate : {round(total_passed / total_evals * 100, 2) if total_evals else 0}%")
    print(f"  Composite mean    : {round(safe_mean(all_composite), 4) if all_composite else 'N/A'}")
    print(f"  Extra metrics     : {list(active_extra_fns.keys()) or 'none'}")
    print(f"  Disabled metrics  : {list(disabled) or 'none'}")
    print(f"  Step 3 time       : {total_time}s")
    print(f"{'=' * 65}")

    return STEP3_PATH


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Dyslexia LLM Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Run all three steps end-to-end
  python run.py all

  # Step 1 only — generate fresh outputs (all templates)
  python run.py generate

  # Step 1 — load from an existing generated file instead of re-generating
  python run.py generate --from-json path/to/existing_step1.json

  # Step 1 — only use the first template (for quick testing)
  python run.py generate --template-slice 0 1

  # Step 2 only — evaluate using default step1 output
  python run.py evaluate

  # Step 2 — evaluate from a custom step1 file
  python run.py evaluate --from-json path/to/step1.json

  # Step 3 only — compute metrics using default step2 output
  python run.py metrics

  # Step 3 — disable specific metrics from scoring
  python run.py metrics --disable rouge1_f1 rouge2_f1 bleu_avg

  # Step 3 — from a custom step2 file
  python run.py metrics --from-json path/to/step2.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    gen = subparsers.add_parser("generate", help="Step 1: generate model outputs")
    gen.add_argument(
        "--from-json",
        metavar="PATH",
        help="Load test cases from an existing JSON file instead of generating",
    )
    gen.add_argument(
        "--template-slice",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only use templates[START:END] (e.g. --template-slice 0 1)",
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    ev = subparsers.add_parser("evaluate", help="Step 2: run judge LLM evaluation")
    ev.add_argument(
        "--from-json",
        metavar="PATH",
        help="Load step 1 output from a custom path instead of outputs/step1_generated.json",
    )

    # ── metrics ───────────────────────────────────────────────────────────────
    met = subparsers.add_parser("metrics", help="Step 3: compute metrics + final report")
    met.add_argument(
        "--from-json",
        metavar="PATH",
        help="Load step 2 output from a custom path instead of outputs/step2_evaluated.json",
    )
    met.add_argument(
        "--disable",
        nargs="+",
        metavar="METRIC",
        help="Metric names to exclude from composite scoring (raw scores still stored)",
    )

    # ── all ───────────────────────────────────────────────────────────────────
    all_cmd = subparsers.add_parser("all", help="Run all three steps end-to-end")
    all_cmd.add_argument(
        "--template-slice",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only use templates[START:END]",
    )
    all_cmd.add_argument(
        "--disable",
        nargs="+",
        metavar="METRIC",
        help="Metrics to disable from step 3 scoring",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        tslice = slice(*args.template_slice) if args.template_slice else slice(None)
        step_generate(from_json=args.from_json, template_slice=tslice)

    elif args.command == "evaluate":
        step_evaluate(from_json=args.from_json)
        
    elif args.command == "metrics":
        step_metrics(
            from_json=args.from_json,
            disable_metrics=args.disable,
            # extra_metrics injected programmatically — see docstring above
        )

    elif args.command == "all":
        tslice = slice(*args.template_slice) if args.template_slice else slice(None)
        step_generate(template_slice=tslice)
        step_evaluate()
        step_metrics(disable_metrics=args.disable)


if __name__ == "__main__":
    main()
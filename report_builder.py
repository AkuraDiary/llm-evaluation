from statistics import safe_mean

# def default_output_builder(ctx: dict) -> dict:
    # return {
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

def simple_output(ctx):
    return {
        "summary": {
            "models": ctx["models"],
            "total_cases": len(ctx["all_results"]),
        }
    }
def build_report(results, grouped_metrics):
    summary = {}

    for metric, scores in grouped_metrics.items():
        summary[metric] = {
            "mean": safe_mean(scores),
            "count": len(scores)
        }

    return {
        "summary": summary,
        "total_cases": len(results),
        "results": results[:5]  # sample only
    }
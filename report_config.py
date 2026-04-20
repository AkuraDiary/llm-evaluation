
from datetime import datetime, timezone
from config import MODELS_TO_EVALUATE, METRICS_CONFIG

def get_experiment_metadata():
    return {
        "experiment_metadata": {
            "experiment_id": "EXP_DYSLEXIA_LLM_EVAL_2024_V3_Q1",
            "title": "Comprehensive Multi-Model Evaluation of LLM-Generated Educational Content for Dyslexic Children",
            "description": (
                "A systematic multi-model evaluation of Large Language Model outputs for generating "
                "educationally appropriate questions and answers tailored to children with dyslexia. "
                "Models evaluated: GPT-4o, GPT-4o-mini, GPT-3.5-turbo. "
                "Evaluation employs six standardized NLP metrics via DeepEval framework covering "
                "relevancy, faithfulness, hallucination, contextual alignment, toxicity, and bias. "
                "Statistical analysis includes non-parametric tests (Mann-Whitney U, Kruskal-Wallis), "
                "95% confidence intervals, bootstrap resampling, Bonferroni correction, "
                "Shapiro-Wilk normality test, and Spearman correlation."
            ),
            "research_domain": "Special Education Technology / Natural Language Processing",
            "task_type": "Educational Question-Answer Generation",
            "target_population": "Children with Dyslexia (Elementary School Level, Grade 1-3)",
            "evaluation_framework": "DeepEval",
            "evaluation_framework_version": DEEPEVAL_VERSION,
            "models_evaluated": MODELS_TO_EVALUATE,
            "llm_backend_judge": "OpenAI GPT (via DeepEval evaluation judge)",
            "timestamp_experiment_start_utc": EXPERIMENT_START_UTC,
            "timestamp_experiment_end_utc": datetime.now(timezone.utc).isoformat(),
            "total_execution_time_seconds": experiment_total_time,
            "python_version": sys.version,
            "platform": platform.platform(),
            "total_test_cases": len(all_test_case_results),
            "total_metrics_per_case": len(METRICS_CONFIG),
            "total_evaluations_performed": len(all_test_case_results) * len(METRICS_CONFIG),
        },
        "evaluation_configuration": {
            "metrics_applied": [
                {
                    "metric_name": cfg["class"].__name__,
                    "threshold": cfg["threshold"],
                    "weight_in_composite_score": cfg["weight"],
                    "inverted_scoring": cfg["class"].__name__ in {"ToxicityMetric", "HallucinationMetric", "BiasMetric"},
                    "description": {
                        "AnswerRelevancyMetric": "Measures how relevant the LLM output is to the input question.",
                        "HallucinationMetric": "Detects fabricated or unsupported information in the output.",
                        "FaithfulnessMetric": "Evaluates consistency of output against provided context.",
                        "ContextualRelevancyMetric": "Assesses how well retrieved context supports the output.",
                        "ToxicityMetric": "Detects harmful, offensive, or inappropriate language.",
                        "BiasMetric": "Identifies discriminatory or unfair language patterns.",
                    }.get(cfg["class"].__name__, ""),
                }
                for cfg in METRICS_CONFIG
            ],
            "system_prompt_used": SYSTEM_PROMPT_DYSLEXIA,
            "composite_score_formula": "Weighted average of adjusted metric scores; inverted metrics use (1 - score)",
            "statistical_methods": [
                "Descriptive statistics (mean, std, median, IQR, skewness, kurtosis)",
                "Shapiro-Wilk normality test (approximation)",
                "Mann-Whitney U test (non-parametric pairwise comparison)",
                "Kruskal-Wallis H test (non-parametric multi-group comparison)",
                "Cohen's d (effect size)",
                "Pearson r + p-value (parametric correlation)",
                "Spearman rho + p-value (non-parametric correlation)",
                "Bonferroni correction (multiple comparisons)",
                "95% Confidence Interval (t-distribution)",
                "Bootstrap CI 95% (1000 resamples, percentile method)",
            ],
        },   
    }

def get_evaluation_config():
    return {
        "metrics": [
            {
                "name": cfg["class"].__name__,
                "threshold": cfg["threshold"],
                "weight": cfg["weight"],
            }
            for cfg in METRICS_CONFIG
        ]
    }

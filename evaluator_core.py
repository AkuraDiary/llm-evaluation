# evaluator.py
from deepeval.test_case import LLMTestCase

def confidence_interval_95(values):
    n = len(values)
    if n < 2:
        return None, None
    mean_val = safe_mean(values)
    std_val = safe_std(values)
    t_critical = 1.96 if n >= 30 else {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    }.get(n, 2.0)
    margin = t_critical * (std_val / math.sqrt(n))
    return round(mean_val - margin, 6), round(mean_val + margin, 6)


def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    corrected_alpha = alpha / n if n > 0 else alpha
    return {
        "original_alpha": alpha,
        "n_comparisons": n,
        "corrected_alpha_bonferroni": round(corrected_alpha, 8),
        "results": [
            {
                "p_value": round(p, 6) if p is not None else None,
                "significant_after_correction": (p is not None and p < corrected_alpha),
            }
            for p in p_values
        ],
    }


def bootstrap_mean_ci(values, n_bootstrap=1000, ci=95, seed=42):
    if len(values) < 2:
        return None, None
    random.seed(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        boot_means.append(safe_mean(sample))
    boot_means.sort()
    lower_idx = int(((100 - ci) / 2 / 100) * n_bootstrap)
    upper_idx = int((1 - (100 - ci) / 2 / 100) * n_bootstrap) - 1
    return round(boot_means[lower_idx], 6), round(boot_means[upper_idx], 6)


def classify_effect_size(d):
    if d is None:
        return "undefined"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def classify_score_quality(score, metric_name):
    inverted_metrics = {"ToxicityMetric", "HallucinationMetric", "BiasMetric"}
    if metric_name in inverted_metrics:
        if score <= 0.1:
            return "Excellent"
        if score <= 0.3:
            return "Good"
        if score <= 0.5:
            return "Acceptable"
        return "Poor"
    else:
        if score >= 0.8:
            return "Excellent"
        if score >= 0.6:
            return "Good"
        if score >= 0.4:
            return "Acceptable"
        return "Poor"


def compute_composite_score(metric_results, weights_map):
    total_weight = 0
    weighted_sum = 0
    inverted_metrics = {"ToxicityMetric", "HallucinationMetric", "BiasMetric"}
    for m in metric_results:
        name = m["metric_name"]
        score = m["score"]
        weight = weights_map.get(name, 1.0)
        adjusted_score = (1.0 - score) if name in inverted_metrics else score
        weighted_sum += adjusted_score * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def evaluate(test_cases, metrics_instances, weights_map):
    return evaluate_test_cases(test_cases, weights_map, metrics_instances)

def evaluate_test_cases(test_cases_data, weights_map, metrics_instances):
    all_results = []

    for tc_data in test_cases_data:
        test_case = LLMTestCase(
            input=tc_data["input"],
            actual_output=tc_data["actual_output"],
            expected_output=tc_data["expected_output"],
            context=tc_data["context"],
            retrieval_context=tc_data["retrieval_context"],
        )

        metrics_evaluation = []

        for m_info in metrics_instances:
            metric = m_info["instance"]

            try:
                metric.measure(test_case)
                score = metric.score or 0.0
            except Exception as e:
                score = 0.0

            metrics_evaluation.append({
                "metric_name": metric.__class__.__name__,
                "score": score,
            })

        composite_score = compute_composite_score(metrics_evaluation, weights_map)

        all_results.append({
            "input": tc_data["input"],
            "output": tc_data["actual_output"],
            "metrics": metrics_evaluation,
            "composite_score": composite_score,
            "model": tc_data["model"]
        })

    return all_results
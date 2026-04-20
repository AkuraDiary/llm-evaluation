import itertools

from config import METRIC_NAMES
from evaluator_core import compute_descriptive
from statistics import (
    pearson_r,
    spearman_rho,
    p_value_from_r,
    mann_whitney_u,
    kruskal_wallis,
    cohens_d,
    bonferroni_correction,
    classify_effect,
    interp_r,
)


def build_academic_report(
    all_test_case_results,
    aggregate_statistics,
    correlation_matrix,
    model_comparison,
    performance_by_category,
    performance_by_difficulty,
    metadata,
    config,
):
    return {
        "experiment_metadata": metadata,
        "evaluation_configuration": config,
        "test_cases": all_test_case_results,
        "aggregate_statistics": aggregate_statistics,
        "metric_correlation_matrix": correlation_matrix,
        "model_comparison_analysis": model_comparison,
        "performance_by_category": performance_by_category,
        "performance_by_difficulty_level": performance_by_difficulty,
    }


def build_aggregate_statistics(raw_by_metric):
    return {m: compute_descriptive(scores) for m, scores in raw_by_metric.items()}


def build_correlation_matrix(raw_by_metric):
    matrix = {}
    names = list(raw_by_metric.keys())
    for m1, m2 in itertools.combinations(names, 2):
        s1, s2 = raw_by_metric[m1], raw_by_metric[m2]
        n = min(len(s1), len(s2))
        pr = pearson_r(s1[:n], s2[:n])
        sr = spearman_rho(s1[:n], s2[:n])
        pp = p_value_from_r(pr, n)
        sp = p_value_from_r(sr, n)
        matrix[f"{m1}_vs_{m2}"] = {
            "metric_a": m1, "metric_b": m2, "n": n,
            "pearson_r": pr, "pearson_p_value": pp,
            "spearman_rho": sr, "spearman_p_value": sp,
            "pearson_significant_p05": (pp < 0.05) if pp is not None else None,
            "spearman_significant_p05": (sp < 0.05) if sp is not None else None,
            "interpretation_pearson": interp_r(pr),
            "interpretation_spearman": interp_r(sr),
        }
    return matrix


def build_model_comparison(by_model, raw_by_metric):
    model_names = list(by_model.keys())
    result = {}
    all_p_values = []

    for m in METRIC_NAMES:
        groups = {mdl: by_model[mdl].get(m, []) for mdl in model_names}
        result[m] = {mdl: compute_descriptive(scores) for mdl, scores in groups.items()}

        pairwise = {}
        for ma, mb in itertools.combinations(model_names, 2):
            u, pu = mann_whitney_u(groups[ma], groups[mb])
            d = cohens_d(groups[ma], groups[mb])
            pairwise[f"{ma}_vs_{mb}"] = {
                "mann_whitney_U": u,
                "p_value": pu,
                "cohens_d": d,
                "effect_size": classify_effect(d),
                "significant_p05": (pu < 0.05) if pu is not None else None,
            }
            if pu is not None:
                all_p_values.append(pu)

        kw_groups = [groups[mdl] for mdl in model_names if groups[mdl]]
        h, kw_p = kruskal_wallis(*kw_groups) if len(kw_groups) >= 2 else (None, None)
        result[m]["statistical_tests"] = {
            "kruskal_wallis_H": h,
            "kruskal_wallis_p_value": kw_p,
            "kruskal_wallis_significant_p05": (kw_p < 0.05) if kw_p is not None else None,
            "pairwise_mann_whitney_u": pairwise,
        }

    return result, bonferroni_correction(all_p_values)


def build_group_analysis(group_dict, group_label):
    result = {}
    group_names = list(group_dict.keys())
    for m in METRIC_NAMES:
        groups_data = {g: group_dict[g].get(m, []) for g in group_names}
        result[m] = {g: compute_descriptive(scores) for g, scores in groups_data.items()}

        pairwise, all_p = {}, []
        for ga, gb in itertools.combinations(group_names, 2):
            u, pu = mann_whitney_u(groups_data[ga], groups_data[gb])
            d = cohens_d(groups_data[ga], groups_data[gb])
            pairwise[f"{ga}_vs_{gb}"] = {
                "mann_whitney_U": u, "p_value": pu,
                "cohens_d": d, "effect_size": classify_effect(d),
                "significant_p05": (pu < 0.05) if pu is not None else None,
            }
            if pu is not None:
                all_p.append(pu)

        kw_gs = [groups_data[g] for g in group_names if groups_data[g]]
        h, kw_p = kruskal_wallis(*kw_gs) if len(kw_gs) >= 2 else (None, None)
        result[m]["statistical_tests"] = {
            "kruskal_wallis_H": h,
            "kruskal_wallis_p_value": kw_p,
            "kruskal_wallis_significant_p05": (kw_p < 0.05) if kw_p is not None else None,
            "pairwise_mann_whitney_u": pairwise,
            "bonferroni_correction": bonferroni_correction(all_p),
        }
    return result

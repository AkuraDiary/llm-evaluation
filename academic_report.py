def build_academic_report(
    all_test_case_results,
    aggregate_statistics,
    correlation_matrix,
    model_comparison,
    performance_by_category,
    performance_by_difficulty,
    metadata,
    config
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
    

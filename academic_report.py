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
    
def export_academic_report(output_document):
    output_path = "output/evaluasi_llm_disleksia_scopus_q1_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_document, f, indent=4, ensure_ascii=False)

    print(f"Hasil disimpan di: {output_path}")
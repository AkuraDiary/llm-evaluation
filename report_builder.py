from statistics import safe_mean

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
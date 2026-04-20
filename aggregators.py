def group_by_model(results):
    grouped = {}

    for r in results:
        model = r["model"]
        grouped.setdefault(model, []).append(r["composite_score"])

    return grouped


def group_by_metric(results):
    grouped = {}

    for r in results:
        for m in r["metrics"]:
            name = m["metric_name"]
            grouped.setdefault(name, []).append(m["score"])

    return grouped
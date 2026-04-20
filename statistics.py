
def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def safe_std(values):
    if len(values) < 2:
        return None
    mean = safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def safe_median(values):
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def safe_percentile(values, p):
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    frac = index - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def cohens_d(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return None
    mean_a = safe_mean(group_a)
    mean_b = safe_mean(group_b)
    std_a = safe_std(group_a)
    std_b = safe_std(group_b)
    pooled_std = math.sqrt(
        ((len(group_a) - 1) * std_a ** 2 + (len(group_b) - 1) * std_b ** 2)
        / (len(group_a) + len(group_b) - 2)
    )
    if pooled_std == 0:
        return None
    return (mean_a - mean_b) / pooled_std


def pearson_correlation(x_vals, y_vals):
    n = len(x_vals)
    if n < 2:
        return None
    mean_x = safe_mean(x_vals)
    mean_y = safe_mean(y_vals)
    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def spearman_correlation(x_vals, y_vals):
    n = len(x_vals)
    if n < 2:
        return None
    rank_x = _rank(x_vals)
    rank_y = _rank(y_vals)
    return pearson_correlation(rank_x, rank_y)


def _rank(vals):
    sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def shapiro_wilk_approx(data):
    n = len(data)
    if n < 3:
        return None, None
    sorted_data = sorted(data)
    mean_val = safe_mean(sorted_data)
    ss = sum((x - mean_val) ** 2 for x in sorted_data)
    if ss == 0:
        return 1.0, 1.0
    m = [(i - (n - 1) / 2) / math.sqrt((n ** 2 - 1) / 12) for i in range(n)]
    b = sum(m[i] * sorted_data[i] for i in range(n))
    w = (b ** 2) / ss
    w = max(0.0, min(1.0, w))
    p_approx = max(0.001, min(0.999, 1.0 - abs(w - 0.95) * 5))
    return round(w, 6), round(p_approx, 6)


def mann_whitney_u(group_a, group_b):
    na = len(group_a)
    nb = len(group_b)
    if na == 0 or nb == 0:
        return None, None
    u1 = sum(1 for a in group_a for b in group_b if a > b) + 0.5 * sum(1 for a in group_a for b in group_b if a == b)
    u2 = na * nb - u1
    u_stat = min(u1, u2)
    mean_u = na * nb / 2.0
    std_u = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if std_u == 0:
        return round(u_stat, 6), None
    z = (u_stat - mean_u) / std_u
    p_approx = 2 * (1 - _norm_cdf(abs(z)))
    return round(u_stat, 6), round(p_approx, 6)


def kruskal_wallis(*groups):
    all_vals = []
    group_indices = []
    for i, g in enumerate(groups):
        for v in g:
            all_vals.append(v)
            group_indices.append(i)
    n = len(all_vals)
    if n < 3:
        return None, None
    sorted_combined = sorted(zip(all_vals, group_indices), key=lambda x: x[0])
    ranks = _rank(all_vals)
    rank_sums = [0.0] * len(groups)
    group_sizes = [0] * len(groups)
    for idx, (val, grp) in enumerate(zip(all_vals, group_indices)):
        rank_sums[grp] += ranks[idx]
        group_sizes[grp] += 1
    h = (12 / (n * (n + 1))) * sum(
        (rank_sums[i] ** 2) / group_sizes[i]
        for i in range(len(groups)) if group_sizes[i] > 0
    ) - 3 * (n + 1)
    df = len(groups) - 1
    p_approx = max(0.001, 1 - _chi2_cdf(h, df)) if h >= 0 else 1.0
    return round(h, 6), round(p_approx, 6)


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _chi2_cdf(x, df):
    if x <= 0:
        return 0.0
    return _regularized_gamma(df / 2, x / 2)


def _regularized_gamma(a, x):
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    MAX_ITER = 200
    TOL = 1e-10
    lngamma_a = math.lgamma(a)
    if x < a + 1:
        ap = a
        delt = s = 1.0 / a
        for _ in range(MAX_ITER):
            ap += 1
            delt *= x / ap
            s += delt
            if abs(delt) < abs(s) * TOL:
                break
        return s * math.exp(-x + a * math.log(x) - lngamma_a)
    else:
        b = x + 1 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, MAX_ITER + 1):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < TOL:
                break
        return 1.0 - math.exp(-x + a * math.log(x) - lngamma_a) * h

import math
import random

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


def pearson_r(x_vals, y_vals):
    x = [v for v in x_vals if v is not None]
    y = [v for v in y_vals if v is not None]
    n = min(len(x), len(y))
    if n < 2:
        return None
    x, y = x[:n], y[:n]
    mx, my = safe_mean(x), safe_mean(y)
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx = math.sqrt(sum((v-mx)**2 for v in x))
    dy = math.sqrt(sum((v-my)**2 for v in y))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 6)


def _rank(vals):
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    sorted_vals = sorted(valid, key=lambda x: x[1])
    ranks = [None] * len(vals)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals)-1 and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x_vals, y_vals):
    rx = _rank(x_vals)
    ry = _rank(y_vals)
    pairs = [(rx[i], ry[i]) for i in range(len(rx)) if rx[i] is not None and ry[i] is not None]
    if len(pairs) < 2:
        return None
    rx2 = [p[0] for p in pairs]
    ry2 = [p[1] for p in pairs]
    return pearson_r(rx2, ry2)


def p_value_from_r(r, n):
    if r is None or n < 3:
        return None
    t = r * math.sqrt(n-2) / math.sqrt(max(1 - r**2, 1e-10))
    return round(2 * (1 - _norm_cdf(abs(t))), 6)


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def shapiro_wilk_approx(data):
    valid = [v for v in data if v is not None]
    n = len(valid)
    if n < 3:
        return None, None
    sd = sorted(valid)
    mv = safe_mean(sd)
    ss = sum((x - mv)**2 for x in sd)
    if ss == 0:
        return 1.0, 1.0
    m = [(i - (n-1)/2) / math.sqrt((n**2-1)/12) for i in range(n)]
    b = sum(m[i]*sd[i] for i in range(n))
    w = max(0.0, min(1.0, (b**2)/ss))
    p = max(0.001, min(0.999, 1.0 - abs(w - 0.95)*5))
    return round(w, 6), round(p, 6)


def mann_whitney_u(a, b):
    ga = [v for v in a if v is not None]
    gb = [v for v in b if v is not None]
    if not ga or not gb:
        return None, None
    u1 = sum(1 for x in ga for y in gb if x > y) + 0.5*sum(1 for x in ga for y in gb if x == y)
    u2 = len(ga)*len(gb) - u1
    u = min(u1, u2)
    mu = len(ga)*len(gb)/2.0
    sigma = math.sqrt(len(ga)*len(gb)*(len(ga)+len(gb)+1)/12.0)
    if sigma == 0:
        return round(u, 6), None
    z = (u - mu) / sigma
    p = round(2*(1 - _norm_cdf(abs(z))), 6)
    return round(u, 6), p


def kruskal_wallis(*groups):
    all_v, grp_idx = [], []
    for i, g in enumerate(groups):
        for v in g:
            if v is not None:
                all_v.append(v)
                grp_idx.append(i)
    n = len(all_v)
    if n < 3:
        return None, None
    ranks = _rank(all_v)
    rs = [0.0]*len(groups)
    ns = [0]*len(groups)
    for idx, gi in enumerate(grp_idx):
        if ranks[idx] is not None:
            rs[gi] += ranks[idx]
            ns[gi] += 1
    h = (12/(n*(n+1))) * sum((rs[i]**2)/ns[i] for i in range(len(groups)) if ns[i] > 0) - 3*(n+1)
    df = len(groups)-1
    p = max(0.001, 1 - _chi2_cdf(h, df)) if h >= 0 else 1.0
    return round(h, 6), round(p, 6)


def _chi2_cdf(x, df):
    if x <= 0:
        return 0.0
    return _reg_gamma(df/2, x/2)


def _reg_gamma(a, x):
    if x <= 0:
        return 0.0
    lg = math.lgamma(a)
    if x < a+1:
        ap, d, s = a, 1.0/a, 1.0/a
        for _ in range(200):
            ap += 1
            d *= x/ap
            s += d
            if abs(d) < abs(s)*1e-10:
                break
        return s * math.exp(-x + a*math.log(x) - lg)
    b, c, d2, h = x+1-a, 1e30, 1/(x+1-a), 1/(x+1-a)
    for i in range(1, 201):
        an = -i*(i-a)
        b += 2
        d2 = an*d2+b
        if abs(d2) < 1e-30:
            d2 = 1e-30
        c = b + an/c
        if abs(c) < 1e-30:
            c = 1e-30
        d2 = 1/d2
        dlt = d2*c
        h *= dlt
        if abs(dlt-1) < 1e-10:
            break
    return 1 - math.exp(-x + a*math.log(x) - lg)*h


def confidence_interval_95(values):
    valid = [v for v in values if v is not None]
    n = len(valid)
    if n < 2:
        return None, None
    mv = safe_mean(valid)
    sv = safe_std(valid)
    if sv is None:
        return None, None
    t_table = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,
               8:2.365,9:2.306,10:2.262,15:2.131,20:2.086,25:2.060,30:2.042}
    tc = 1.96 if n >= 30 else t_table.get(n, 2.0)
    margin = tc * (sv / math.sqrt(n))
    return round(mv - margin, 6), round(mv + margin, 6)


def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    ca = alpha/n if n > 0 else alpha
    return {
        "original_alpha": alpha,
        "n_comparisons": n,
        "corrected_alpha": round(ca, 8),
        "results": [
            {"p_value": round(p,6) if p is not None else None,
             "significant_after_correction": (p is not None and p < ca)}
            for p in p_values
        ],
    }


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None, None
    random.seed(seed)
    n = len(valid)
    boot_means = sorted([safe_mean([random.choice(valid) for _ in range(n)]) for _ in range(n_boot)])
    li = int(((100-ci)/2/100)*n_boot)
    ui = int((1-(100-ci)/2/100)*n_boot)-1
    return round(boot_means[li], 6), round(boot_means[ui], 6)


def classify_effect(d):
    if d is None:
        return "undefined"
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def interp_r(r):
    if r is None: return "undefined"
    if r >= 0.7: return "strong_positive"
    if r >= 0.4: return "moderate_positive"
    if r > 0: return "weak_positive"
    if r <= -0.7: return "strong_negative"
    if r <= -0.4: return "moderate_negative"
    if r < 0: return "weak_negative"
    return "no_correlation"


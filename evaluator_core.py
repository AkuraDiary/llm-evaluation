import math
import re
import time
from collections import Counter

from config import (
    JUDGE_MODEL,
    METRIC_NAMES,
    METRIC_WEIGHTS,
    INVERTED_METRICS,
    TOXIC_WORDS_ID,
    BIAS_WORDS_ID,
    COMPLEX_WORDS_ID,
)
from statistics import (
    safe_mean,
    safe_std,
    safe_median,
    safe_percentile,
    shapiro_wilk_approx,
    confidence_interval_95,
    bootstrap_ci,
    bonferroni_correction,
    mann_whitney_u,
    kruskal_wallis,
    cohens_d,
    pearson_r,
    spearman_rho,
    p_value_from_r,
    classify_effect,
    interp_r,
)
from util import tokenize


# ── NLP similarity ────────────────────────────────────────────────────────────

def compute_rouge_scores(hypothesis, reference):
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    zero = {
        "rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0,
        "rouge2_precision": 0.0, "rouge2_recall": 0.0, "rouge2_f1": 0.0,
        "rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0,
    }
    if not hyp_tokens or not ref_tokens:
        return zero

    def ngrams(tokens, n):
        return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])

    def prf(hyp_ng, ref_ng):
        overlap = sum((hyp_ng & ref_ng).values())
        prec = overlap / sum(hyp_ng.values()) if hyp_ng else 0.0
        rec = overlap / sum(ref_ng.values()) if ref_ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 6), round(rec, 6), round(f1, 6)

    r1p, r1r, r1f = prf(ngrams(hyp_tokens, 1), ngrams(ref_tokens, 1))
    r2p, r2r, r2f = prf(ngrams(hyp_tokens, 2), ngrams(ref_tokens, 2))

    def lcs_len(x, y):
        m, n = len(x), len(y)
        if not m or not n:
            return 0
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]

    lcs = lcs_len(hyp_tokens, ref_tokens)
    rlp = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    rlr = lcs / len(ref_tokens) if ref_tokens else 0.0
    rlf = 2 * rlp * rlr / (rlp + rlr) if (rlp + rlr) > 0 else 0.0

    return {
        "rouge1_precision": r1p, "rouge1_recall": r1r, "rouge1_f1": r1f,
        "rouge2_precision": r2p, "rouge2_recall": r2r, "rouge2_f1": r2f,
        "rougeL_precision": round(rlp, 6), "rougeL_recall": round(rlr, 6), "rougeL_f1": round(rlf, 6),
    }


def compute_bleu_score(hypothesis, reference):
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    if not hyp_tokens or not ref_tokens:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu_avg": 0.0}

    def ngram_precision(hyp, ref, n):
        hyp_ng = Counter([tuple(hyp[i:i + n]) for i in range(max(0, len(hyp) - n + 1))])
        ref_ng = Counter([tuple(ref[i:i + n]) for i in range(max(0, len(ref) - n + 1))])
        if not hyp_ng:
            return 0.0
        return sum((hyp_ng & ref_ng).values()) / sum(hyp_ng.values())

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens)) if len(hyp_tokens) > 0 else 0.0

    precisions = [ngram_precision(hyp_tokens, ref_tokens, n) for n in range(1, 5)]
    bleu_scores = {}
    for i, p in enumerate(precisions, 1):
        bleu_scores[f"bleu{i}"] = round(bp * math.exp(math.log(p)) if p > 0 else 0.0, 6)

    valid_logs = [math.log(p) for p in precisions if p > 0]
    bleu_scores["bleu_avg"] = round(bp * math.exp(sum(valid_logs) / len(valid_logs)) if valid_logs else 0.0, 6)
    return bleu_scores


# ── Readability ───────────────────────────────────────────────────────────────

def compute_readability_metrics(text):
    """Return a comprehensive readability dict for the given text."""
    empty = {
        "total_chars": 0, "total_words": 0, "total_sentences": 0,
        "total_syllables": 0, "avg_word_length_chars": 0.0,
        "avg_sentence_length_words": 0.0, "avg_syllables_per_word": 0.0,
        "flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0,
        "gunning_fog_index": 0.0, "smog_index": 0.0,
        "automated_readability_index": 0.0, "coleman_liau_index": 0.0,
        "type_token_ratio": 0.0, "lexical_diversity": 0.0,
        "max_sentence_length_words": 0, "min_sentence_length_words": 0,
        "sentences_exceeding_8_words": 0, "sentences_exceeding_8_words_pct": 0.0,
        "complex_word_count_in_forbidden_list": 0, "complex_word_ratio_forbidden": 0.0,
        "has_explicit_answer": False, "has_question_mark": False,
        "numeric_count": 0, "numbers_in_range_1_20": True,
    }
    if not text or not text.strip():
        return empty

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    words = tokenize(text)
    n_words = len(words)
    n_chars = len(text.replace(' ', ''))

    def count_syllables_id(word):
        vowels = 'aiueoáéíóú'
        count, prev_vowel = 0, False
        for ch in word.lower():
            is_v = ch in vowels
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        return max(1, count)

    syllables_per_word = [count_syllables_id(w) for w in words]
    n_syllables = sum(syllables_per_word)

    avg_word_len = n_chars / n_words if n_words > 0 else 0.0
    avg_sent_len = n_words / n_sentences
    avg_syl_per_word = n_syllables / n_words if n_words > 0 else 0.0

    flesch = max(0.0, min(100.0, 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syl_per_word))
    fk_grade = 0.39 * avg_sent_len + 11.8 * avg_syl_per_word - 15.59

    complex_words = [w for w in words if count_syllables_id(w) >= 3]
    n_complex = len(complex_words)
    fog = 0.4 * (avg_sent_len + 100 * n_complex / n_words) if n_words > 0 else 0.0
    smog = 3 + math.sqrt(n_complex * (30 / n_sentences)) if n_sentences > 0 else 0.0
    ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43 if n_words > 0 else 0.0
    L = (n_chars / n_words) * 100 if n_words > 0 else 0.0
    S = (n_sentences / n_words) * 100 if n_words > 0 else 0.0
    cli = 0.0588 * L - 0.296 * S - 15.8

    ttr = len(set(words)) / n_words if n_words > 0 else 0.0
    sent_lengths = [len(tokenize(s)) for s in sentences]
    sentences_over_8 = sum(1 for sl in sent_lengths if sl > 8)
    pct_over_8 = sentences_over_8 / n_sentences if n_sentences > 0 else 0.0

    bad_complex = [w for w in words if w in COMPLEX_WORDS_ID]
    bad_complex_ratio = len(bad_complex) / n_words if n_words > 0 else 0.0

    has_answer = bool(re.search(r'jawaban|answer|=\s*\d', text.lower()))
    numbers_found = re.findall(r'\b\d+\b', text)
    numbers_ok = all(1 <= int(n) <= 20 for n in numbers_found) if numbers_found else True

    return {
        "total_chars": n_chars,
        "total_words": n_words,
        "total_sentences": n_sentences,
        "total_syllables": n_syllables,
        "avg_word_length_chars": round(avg_word_len, 4),
        "avg_sentence_length_words": round(avg_sent_len, 4),
        "avg_syllables_per_word": round(avg_syl_per_word, 4),
        "flesch_reading_ease": round(flesch, 4),
        "flesch_kincaid_grade": round(fk_grade, 4),
        "gunning_fog_index": round(fog, 4),
        "smog_index": round(smog, 4),
        "automated_readability_index": round(ari, 4),
        "coleman_liau_index": round(cli, 4),
        "type_token_ratio": round(ttr, 6),
        "lexical_diversity": round(ttr, 6),
        "max_sentence_length_words": max(sent_lengths) if sent_lengths else 0,
        "min_sentence_length_words": min(sent_lengths) if sent_lengths else 0,
        "sentences_exceeding_8_words": sentences_over_8,
        "sentences_exceeding_8_words_pct": round(pct_over_8, 6),
        "complex_word_count_in_forbidden_list": len(bad_complex),
        "complex_word_ratio_forbidden": round(bad_complex_ratio, 6),
        "has_explicit_answer": has_answer,
        "has_question_mark": '?' in text,
        "numeric_count": len(numbers_found),
        "numbers_in_range_1_20": numbers_ok,
    }


def compute_dyslexia_compliance_score(readability):
    """Weighted rubric: sentence length (30%), Flesch RE (25%), explicit answer (20%),
    number range (15%), forbidden words (10%)."""
    score, max_score = 0.0, 0.0

    max_score += 30.0
    asl = readability["avg_sentence_length_words"]
    if asl <= 6:
        score += 30.0
    elif asl <= 8:
        score += 20.0
    elif asl <= 10:
        score += 10.0

    max_score += 25.0
    fre = readability["flesch_reading_ease"]
    if fre >= 80:
        score += 25.0
    elif fre >= 60:
        score += 18.0
    elif fre >= 40:
        score += 10.0

    max_score += 20.0
    if readability["has_explicit_answer"]:
        score += 20.0

    max_score += 15.0
    if readability["numbers_in_range_1_20"]:
        score += 15.0

    max_score += 10.0
    bad = readability["complex_word_count_in_forbidden_list"]
    if bad == 0:
        score += 10.0
    elif bad <= 1:
        score += 5.0

    return round(score / max_score, 6) if max_score > 0 else 0.0


# ── Safety / fairness ─────────────────────────────────────────────────────────

def compute_toxicity_score(text):
    text_lower = text.lower()
    found = [w for w in TOXIC_WORDS_ID if w in text_lower]
    if not found:
        return 0.0, []
    return round(min(1.0, len(found) / 3.0), 6), found


def compute_bias_score(text):
    text_lower = text.lower()
    found = [p for p in BIAS_WORDS_ID if p in text_lower]
    if not found:
        return 0.0, []
    return round(min(1.0, len(found) / 2.0), 6), found


# ── Relevance & faithfulness ─────────────────────────────────────────────────

def compute_context_relevancy(actual_output, context_list):
    if not context_list or not actual_output:
        return 0.0
    total = sum(compute_rouge_scores(actual_output, ctx)["rouge1_f1"] for ctx in context_list)
    return round(min(1.0, (total / len(context_list)) * 2.5), 6)


def compute_faithfulness_score(actual_output, context_list):
    if not context_list or not actual_output:
        return 0.0
    hyp_tokens = set(tokenize(actual_output))
    ctx_tokens = set()
    for ctx in context_list:
        ctx_tokens.update(tokenize(ctx))
    if not hyp_tokens:
        return 0.0
    overlap = len(hyp_tokens & ctx_tokens)
    return round(min(1.0, (overlap / len(hyp_tokens)) * 1.5), 6)


def compute_answer_relevancy(actual_output, input_prompt):
    if not actual_output or not input_prompt:
        return 0.0
    rouge = compute_rouge_scores(actual_output, input_prompt)
    bleu = compute_bleu_score(actual_output, input_prompt)
    combined = rouge["rouge1_f1"] * 0.4 + rouge["rougeL_f1"] * 0.3 + bleu["bleu1"] * 0.3
    return round(min(1.0, combined * 3.0), 6)


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

def llm_judge_score(ollama_client, actual_output, input_prompt, context_list, expected_output):
    context_str = "\n".join(f"- {c}" for c in context_list)
    judge_prompt = (
        f"Kamu adalah evaluator pendidikan yang menilai kualitas soal untuk anak disleksia.\n\n"
        f"INPUT PROMPT: {input_prompt}\n"
        f"EXPECTED OUTPUT: {expected_output}\n"
        f"ACTUAL OUTPUT: {actual_output}\n"
        f"CONTEXT GUIDELINES:\n{context_str}\n\n"
        f"Nilai output di atas dari skala 0.0 hingga 1.0 berdasarkan:\n"
        f"1. Kesesuaian dengan panduan disleksia (kalimat pendek, kata sederhana)\n"
        f"2. Relevansi dengan prompt\n"
        f"3. Kelengkapan jawaban\n"
        f"4. Ketepatan konten edukatif\n\n"
        f"Berikan HANYA satu angka desimal antara 0.0 dan 1.0. Tidak perlu penjelasan.\n"
        f"Score:"
    )
    try:
        response = ollama_client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            options={"temperature": 0.1, "num_predict": 10},
        )
        raw = (response["message"]["content"] if isinstance(response, dict)
               else response.message.content).strip()
        numbers = re.findall(r'\d+\.?\d*', raw)
        if numbers:
            score = float(numbers[0])
            if score > 1.0:
                score /= 10.0
            return round(min(1.0, max(0.0, score)), 6), raw
        return 0.5, raw
    except Exception as e:
        return 0.5, f"[JUDGE ERROR: {str(e)}]"


# ── Scoring helpers ───────────────────────────────────────────────────────────

def normalize_metric(name, value):
    if name == "flesch_reading_ease":
        return round(min(1.0, max(0.0, value / 100.0)), 6)
    if name == "avg_sentence_length_words":
        if value <= 6:  return 1.0
        if value <= 8:  return 0.75
        if value <= 12: return 0.5
        return 0.2
    if name in INVERTED_METRICS and name != "avg_sentence_length_words":
        return round(1.0 - min(1.0, value), 6)
    return round(min(1.0, max(0.0, value)), 6)


def compute_composite(metric_dict):
    total_w, weighted_sum = 0.0, 0.0
    for name, value in metric_dict.items():
        if value is None:
            continue
        w = METRIC_WEIGHTS.get(name, 1.0)
        weighted_sum += normalize_metric(name, value) * w
        total_w += w
    return round(weighted_sum / total_w, 6) if total_w > 0 else None


def compute_descriptive(scores):
    valid = [v for v in scores if v is not None]
    n = len(valid)
    if n == 0:
        return {"n": 0}
    mv = safe_mean(valid)
    sv = safe_std(valid)
    med = safe_median(valid)
    q1 = safe_percentile(valid, 25)
    q3 = safe_percentile(valid, 75)
    iqr = round(q3 - q1, 6) if q1 is not None and q3 is not None else None
    cv = round(sv / mv, 6) if sv is not None and mv and mv != 0 else None
    skew = None
    if n >= 3 and sv and sv > 0:
        skew = round((n / ((n - 1) * (n - 2))) * sum(((x - mv) / sv) ** 3 for x in valid), 6)
    kurt = None
    if n >= 4 and sv and sv > 0:
        kurt = round(
            (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mv) / sv) ** 4 for x in valid)
            - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3))), 6
        )
    w_stat, p_sw = shapiro_wilk_approx(valid)
    ci_lo, ci_hi = confidence_interval_95(valid)
    bci_lo, bci_hi = bootstrap_ci(valid)
    return {
        "n": n,
        "mean": round(mv, 6) if mv is not None else None,
        "std": round(sv, 6) if sv is not None else None,
        "median": round(med, 6) if med is not None else None,
        "min": round(min(valid), 6),
        "max": round(max(valid), 6),
        "range": round(max(valid) - min(valid), 6),
        "q1_25th": round(q1, 6) if q1 is not None else None,
        "q3_75th": round(q3, 6) if q3 is not None else None,
        "iqr": iqr,
        "coefficient_of_variation": cv,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "shapiro_wilk": {
            "W": w_stat,
            "p_value_approx": p_sw,
            "normally_distributed_p05": (p_sw > 0.05) if p_sw is not None else None,
            "note": "Approximation only. Use scipy.stats.shapiro for exact test.",
        },
        "confidence_interval_95pct_t": {"lower": ci_lo, "upper": ci_hi},
        "bootstrap_ci_95pct_2000_resamples": {"lower": bci_lo, "upper": bci_hi},
        "all_scores": valid,
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_all(test_cases_data, ollama_client):
    """
    Evaluate every test case with all 14 custom metrics.

    Returns:
        all_results     – list of rich per-test-case result dicts
        raw_by_metric   – {metric_name: [scores]}
        by_category     – {category: {metric_name: [scores]}}
        by_difficulty   – {difficulty: {metric_name: [scores]}}
        by_model        – {model: {metric_name: [scores]}}
    """
    all_results = []
    raw_by_metric = {m: [] for m in METRIC_NAMES}
    by_category, by_difficulty, by_model = {}, {}, {}

    total = len(test_cases_data)
    for idx, tc in enumerate(test_cases_data):
        t0 = time.time()
        print(f"  [{idx + 1}/{total}] Evaluating: {tc['id']}")

        rouge = compute_rouge_scores(tc["actual_output"], tc["expected_output"])
        bleu = compute_bleu_score(tc["actual_output"], tc["expected_output"])
        read = compute_readability_metrics(tc["actual_output"])
        dyslexia_score = compute_dyslexia_compliance_score(read)
        toxicity_score, toxic_words_found = compute_toxicity_score(tc["actual_output"])
        bias_score, bias_phrases_found = compute_bias_score(tc["actual_output"])
        ctx_rel = compute_context_relevancy(tc["actual_output"], tc["context"])
        faith = compute_faithfulness_score(tc["actual_output"], tc["context"])
        ans_rel = compute_answer_relevancy(tc["actual_output"], tc["input"])
        judge_score, judge_raw = llm_judge_score(
            ollama_client, tc["actual_output"], tc["input"],
            tc["context"], tc["expected_output"]
        )

        metric_values = {
            "answer_relevancy": ans_rel,
            "context_relevancy": ctx_rel,
            "faithfulness": faith,
            "toxicity": toxicity_score,
            "bias": bias_score,
            "dyslexia_compliance": dyslexia_score,
            "llm_judge_score": judge_score,
            "rouge1_f1": rouge["rouge1_f1"],
            "rouge2_f1": rouge["rouge2_f1"],
            "rougeL_f1": rouge["rougeL_f1"],
            "bleu_avg": bleu["bleu_avg"],
            "flesch_reading_ease": read["flesch_reading_ease"],
            "avg_sentence_length_words": read["avg_sentence_length_words"],
            "type_token_ratio": read["type_token_ratio"],
        }

        composite = compute_composite(metric_values)

        for m, v in metric_values.items():
            raw_by_metric[m].append(v)

        for store, key in [
            (by_category, tc["category"]),
            (by_difficulty, tc["difficulty_level"]),
            (by_model, tc["model"]),
        ]:
            if key not in store:
                store[key] = {m: [] for m in METRIC_NAMES}
            for m, v in metric_values.items():
                store[key][m].append(v)

        tc_time = round(time.time() - t0, 4)

        metrics_detail = []
        for m, v in metric_values.items():
            norm_v = normalize_metric(m, v)
            w = METRIC_WEIGHTS.get(m, 1.0)
            threshold = 0.1 if m in {"toxicity", "bias"} else 0.5
            is_inverted = m in INVERTED_METRICS
            passed = (v <= threshold) if is_inverted else (v >= threshold)
            metrics_detail.append({
                "metric_name": m,
                "raw_score": round(v, 6) if v is not None else None,
                "normalized_score": norm_v,
                "weight": w,
                "threshold": threshold,
                "is_inverted_metric": is_inverted,
                "passed_threshold": passed,
                "weighted_contribution": round(norm_v * w, 6),
            })

        passed_count = sum(1 for m in metrics_detail if m["passed_threshold"])

        all_results.append({
            "test_case_id": tc["id"],
            "category": tc["category"],
            "sub_category": tc["sub_category"],
            "difficulty_level": tc["difficulty_level"],
            "model": tc["model"],
            "input_prompt": tc["input"],
            "actual_output": tc["actual_output"],
            "expected_output": tc["expected_output"],
            "context_provided": tc["context"],
            "metrics_detail": metrics_detail,
            "rouge_scores_full": rouge,
            "bleu_scores_full": bleu,
            "readability_analysis": read,
            "toxicity_analysis": {"score": toxicity_score, "toxic_words_found": toxic_words_found},
            "bias_analysis": {"score": bias_score, "bias_phrases_found": bias_phrases_found},
            "llm_judge": {"score": judge_score, "raw_response": judge_raw, "judge_model": JUDGE_MODEL},
            "test_case_summary": {
                "composite_weighted_score": composite,
                "total_metrics": len(metrics_detail),
                "metrics_passed": passed_count,
                "metrics_failed": len(metrics_detail) - passed_count,
                "pass_rate": round(passed_count / len(metrics_detail), 6),
                "dyslexia_compliance_score": dyslexia_score,
                "execution_time_seconds": tc_time,
            },
        })

    return all_results, raw_by_metric, by_category, by_difficulty, by_model

import json
import os
import time
import math
import platform
import sys
import itertools
import random
from datetime import datetime, timezone
from openai import OpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ToxicityMetric,
    BiasMetric,
)
from ollama import Client

os.environ["OPENAI_API_KEY"] = ""
os.environ["OLLAMA_API_KEY"] = ""

MODELS_TO_EVALUATE = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

DEEPEVAL_VERSION = "unknown"
try:
    import deepeval
    DEEPEVAL_VERSION = getattr(deepeval, "__version__", "unknown")
except Exception:
    pass

EXPERIMENT_START_UTC = datetime.now(timezone.utc).isoformat()

SYSTEM_PROMPT_DYSLEXIA = (
    "Kamu adalah guru sekolah dasar yang ahli dalam membuat soal untuk anak-anak dengan disleksia. "
    "Ikuti aturan berikut dengan ketat:\n"
    "1. Gunakan kalimat sangat pendek, maksimal 8 kata per kalimat.\n"
    "2. Gunakan kata-kata sederhana sehari-hari, hindari istilah teknis.\n"
    "3. Gunakan objek konkret yang mudah divisualisasikan.\n"
    "4. Angka harus kecil, antara 1 sampai 20.\n"
    "5. Sertakan jawaban secara eksplisit di akhir.\n"
    "6. Jangan gunakan kalimat majemuk bertingkat."
)

TEMPLATE_TEST_CASES = [
    {
        "id_prefix": "MAT_PENJUMLAHAN",
        "category": "Matematika",
        "sub_category": "Penjumlahan",
        "difficulty_level": "Mudah",
        "expected_output": "Soal penjumlahan dengan kalimat sangat pendek, objek konkret, angka 1-10, dan jawaban eksplisit.",
        "context": [
            "Anak disleksia membutuhkan kalimat pendek maksimal 8 kata per kalimat.",
            "Gunakan benda konkret yang dapat divisualisasikan.",
            "Hindari kalimat majemuk dan bertumpuk.",
            "Angka harus kecil dan mudah dibayangkan antara 1 sampai 10.",
        ],
        "prompts": [
            "Buatkan 1 soal matematika penjumlahan sederhana untuk anak disleksia dengan tema buah apel beserta jawabannya.",
            "Buatkan 1 soal penjumlahan untuk anak disleksia bertema bola mainan beserta jawabannya.",
            "Buatkan soal penjumlahan sederhana bertema kelereng untuk anak disleksia kelas 1 SD.",
            "Buat soal penjumlahan anak disleksia bertema pensil dengan jawaban yang jelas.",
            "Buat soal penjumlahan untuk anak disleksia dengan tema ikan di akuarium.",
            "Buatkan soal penjumlahan bertema bunga di taman untuk anak disleksia.",
            "Buatkan soal penjumlahan sederhana bertema kucing dan anjing untuk anak disleksia.",
            "Buat 1 soal penjumlahan untuk anak disleksia bertema permen dengan jawaban.",
        ],
    },
    {
        "id_prefix": "MAT_PENGURANGAN",
        "category": "Matematika",
        "sub_category": "Pengurangan",
        "difficulty_level": "Mudah",
        "expected_output": "Soal pengurangan dengan konteks konkret, kalimat pendek, angka 1-10, dan jawaban jelas.",
        "context": [
            "Gunakan cerita konkret dalam kehidupan sehari-hari.",
            "Kalimat tidak boleh lebih dari 8 kata.",
            "Angka antara 1 sampai 10 yang mudah dibayangkan.",
            "Jawaban harus ditampilkan secara eksplisit dengan persamaan.",
        ],
        "prompts": [
            "Buatkan 1 soal pengurangan sederhana untuk anak disleksia bertema buah jeruk beserta jawabannya.",
            "Buat soal pengurangan untuk anak disleksia dengan tema buku.",
            "Buatkan soal pengurangan bertema kue untuk anak disleksia kelas 2 SD.",
            "Buat soal pengurangan sederhana bertema telur ayam untuk anak disleksia.",
            "Buatkan soal pengurangan untuk anak disleksia dengan tema balon ulang tahun.",
            "Buat 1 soal pengurangan bertema burung di pohon untuk anak disleksia.",
        ],
    },
    {
        "id_prefix": "MAT_PERKALIAN",
        "category": "Matematika",
        "sub_category": "Perkalian",
        "difficulty_level": "Sedang",
        "expected_output": "Soal perkalian dengan konteks konkret, petunjuk cara menjawab, dan persamaan matematis eksplisit.",
        "context": [
            "Menggunakan konteks konkret membuat soal lebih mudah dipahami.",
            "Petunjuk cara menghitung sangat membantu anak disleksia.",
            "Kalimat tetap pendek dan menggunakan kata sehari-hari.",
            "Jawaban harus menampilkan persamaan matematika secara eksplisit.",
        ],
        "prompts": [
            "Buatkan soal perkalian untuk anak disleksia menggunakan tema jari tangan.",
            "Buat soal perkalian sederhana bertema roda sepeda untuk anak disleksia.",
            "Buatkan soal perkalian untuk anak disleksia dengan tema kotak pensil.",
            "Buat soal perkalian bertema telur dalam keranjang untuk anak disleksia.",
            "Buatkan soal perkalian bertema kaki binatang untuk anak disleksia kelas 3 SD.",
            "Buat soal perkalian untuk anak disleksia dengan tema kantong permen.",
        ],
    },
    {
        "id_prefix": "MAT_PEMBAGIAN",
        "category": "Matematika",
        "sub_category": "Pembagian",
        "difficulty_level": "Sedang",
        "expected_output": "Soal pembagian singkat dengan bahasa sehari-hari, kalimat tidak lebih dari 8 kata, dan jawaban eksplisit.",
        "context": [
            "Kosa kata harus sangat sederhana dan mudah dimengerti anak SD.",
            "Hindari kalimat panjang yang penuh deskripsi tidak penting.",
            "Hindari kata teknis.",
            "Fokus pada soal utama tanpa cerita latar berlebihan.",
        ],
        "prompts": [
            "Buatkan soal cerita pembagian untuk anak disleksia dengan tema cokelat.",
            "Buat soal pembagian sederhana bertema apel dibagi adik untuk anak disleksia.",
            "Buatkan soal pembagian untuk anak disleksia dengan tema roti.",
            "Buat soal pembagian bertema kelereng dibagi teman untuk anak disleksia.",
            "Buatkan soal pembagian sederhana bertema pizza untuk anak disleksia kelas 3 SD.",
            "Buat soal pembagian untuk anak disleksia dengan tema bunga dibagi vas.",
        ],
    },
    {
        "id_prefix": "IND_MEMBACA",
        "category": "Bahasa Indonesia",
        "sub_category": "Membaca",
        "difficulty_level": "Mudah",
        "expected_output": "Teks bacaan 2-3 kalimat pendek, satu pertanyaan literal, dan jawaban langsung dari teks.",
        "context": [
            "Teks bacaan untuk disleksia maksimal 3 kalimat pendek.",
            "Setiap kalimat tidak lebih dari 6 kata.",
            "Pertanyaan harus mengacu langsung pada informasi eksplisit dalam teks.",
            "Jawaban harus tersedia jelas dalam teks sumber.",
        ],
        "prompts": [
            "Buatkan 1 soal membaca pemahaman singkat untuk anak disleksia kelas 2 SD beserta jawaban.",
            "Buat teks bacaan pendek dan soal pemahaman untuk anak disleksia bertema hewan peliharaan.",
            "Buatkan soal membaca pemahaman singkat untuk anak disleksia bertema sekolah.",
            "Buat teks bacaan 2-3 kalimat dan 1 pertanyaan untuk anak disleksia bertema buah.",
            "Buatkan soal membaca pemahaman untuk anak disleksia dengan teks tentang cuaca.",
            "Buat soal pemahaman bacaan singkat untuk anak disleksia bertema keluarga.",
            "Buatkan teks pendek dan soal literal untuk anak disleksia bertema makanan.",
        ],
    },
    {
        "id_prefix": "IND_MENULIS",
        "category": "Bahasa Indonesia",
        "sub_category": "Menulis",
        "difficulty_level": "Sedang",
        "expected_output": "Soal kalimat rumpang dengan pilihan kata tersedia, kalimat sederhana, dan jawaban logis.",
        "context": [
            "Format kalimat rumpang lebih mudah bagi anak disleksia daripada menulis bebas.",
            "Menyediakan pilihan kata mengurangi tekanan kognitif.",
            "Kata pilihan harus sangat berbeda secara semantik.",
            "Konteks kalimat harus dekat dengan pengalaman sehari-hari anak.",
        ],
        "prompts": [
            "Buatkan panduan soal menulis kalimat sederhana untuk anak disleksia kelas 3 SD beserta contoh jawaban.",
            "Buat soal melengkapi kalimat rumpang untuk anak disleksia dengan pilihan kata.",
            "Buatkan 3 soal kalimat rumpang sederhana untuk anak disleksia beserta pilihan kata dan jawaban.",
            "Buat soal isi kalimat rumpang untuk anak disleksia bertema aktivitas sehari-hari.",
            "Buatkan soal menulis untuk anak disleksia dengan format pilih kata yang tepat.",
            "Buat soal melengkapi kalimat untuk anak disleksia dengan tema hobi.",
        ],
    },
    {
        "id_prefix": "IPA_HEWAN",
        "category": "IPA",
        "sub_category": "Hewan",
        "difficulty_level": "Mudah",
        "expected_output": "Soal pilihan ganda singkat, kalimat tanya pendek maksimal 5 kata, dan pilihan jawaban kontras.",
        "context": [
            "Format pilihan ganda sangat membantu anak disleksia.",
            "Kalimat pertanyaan tidak lebih dari 5 kata.",
            "Pilihan jawaban harus kontras dan jelas berbeda.",
            "Gunakan topik yang familiar dengan kehidupan sehari-hari anak.",
        ],
        "prompts": [
            "Buatkan 1 soal IPA sederhana tentang hewan untuk anak disleksia kelas 3 SD beserta jawaban.",
            "Buat soal pilihan ganda IPA tentang makanan hewan untuk anak disleksia.",
            "Buatkan soal IPA tentang habitat hewan untuk anak disleksia dengan format pilihan ganda.",
            "Buat soal pilihan ganda tentang ciri hewan untuk anak disleksia kelas 2 SD.",
            "Buatkan soal IPA sederhana tentang cara gerak hewan untuk anak disleksia.",
            "Buat soal pilihan ganda tentang hewan berkaki empat untuk anak disleksia.",
            "Buatkan soal IPA tentang suara hewan untuk anak disleksia kelas 1 SD.",
        ],
    },
    {
        "id_prefix": "IPA_TUMBUHAN",
        "category": "IPA",
        "sub_category": "Tumbuhan",
        "difficulty_level": "Mudah",
        "expected_output": "Soal pilihan ganda tentang tumbuhan dengan kalimat pendek dan pilihan jawaban jelas.",
        "context": [
            "Gunakan analogi sederhana untuk menjelaskan konsep tumbuhan.",
            "Hindari terminologi ilmiah kompleks.",
            "Format pilihan ganda sangat disarankan untuk anak disleksia.",
            "Pertanyaan harus dijawab dengan satu kata atau kalimat pendek.",
        ],
        "prompts": [
            "Buatkan soal IPA sederhana tentang bagian tumbuhan untuk anak disleksia kelas 2 SD.",
            "Buat soal pilihan ganda tentang kebutuhan tumbuhan untuk anak disleksia.",
            "Buatkan soal IPA tentang warna daun untuk anak disleksia dengan pilihan ganda.",
            "Buat soal sederhana tentang buah dan biji untuk anak disleksia kelas 3 SD.",
            "Buatkan soal pilihan ganda tentang akar tumbuhan untuk anak disleksia.",
            "Buat soal IPA tentang tumbuhan yang dimakan untuk anak disleksia.",
        ],
    },
    {
        "id_prefix": "IPS_LINGKUNGAN",
        "category": "IPS",
        "sub_category": "Lingkungan",
        "difficulty_level": "Mudah",
        "expected_output": "Soal IPS tentang lingkungan sekitar dengan bahasa sangat sederhana dan jawaban konkret.",
        "context": [
            "Gunakan konteks lingkungan yang dikenal anak sehari-hari.",
            "Kalimat sangat pendek dan tidak bertingkat.",
            "Pilihan jawaban untuk soal pilihan ganda harus sangat kontras.",
            "Hindari konsep abstrak seperti norma dan nilai sosial.",
        ],
        "prompts": [
            "Buatkan soal IPS sederhana tentang lingkungan rumah untuk anak disleksia kelas 2 SD.",
            "Buat soal pilihan ganda IPS tentang kebersihan lingkungan untuk anak disleksia.",
            "Buatkan soal IPS tentang tempat-tempat di sekitar sekolah untuk anak disleksia.",
            "Buat soal sederhana tentang gotong royong untuk anak disleksia kelas 3 SD.",
            "Buatkan soal IPS tentang profesi di lingkungan sekitar untuk anak disleksia.",
        ],
    },
    {
        "id_prefix": "IPS_KELUARGA",
        "category": "IPS",
        "sub_category": "Keluarga",
        "difficulty_level": "Mudah",
        "expected_output": "Soal IPS tentang keluarga dengan kalimat pendek, pilihan ganda, dan jawaban konkret.",
        "context": [
            "Tema keluarga dekat dengan kehidupan sehari-hari anak.",
            "Kalimat pendek maksimal 6 kata.",
            "Gunakan format pilihan ganda untuk mengurangi beban menulis.",
            "Hindari pertanyaan yang membutuhkan penalaran abstrak.",
        ],
        "prompts": [
            "Buatkan soal IPS tentang anggota keluarga untuk anak disleksia kelas 1 SD.",
            "Buat soal pilihan ganda IPS tentang peran ayah dan ibu untuk anak disleksia.",
            "Buatkan soal sederhana tentang kegiatan keluarga untuk anak disleksia.",
            "Buat soal IPS tentang silsilah keluarga sederhana untuk anak disleksia kelas 2 SD.",
            "Buatkan soal pilihan ganda tentang kasih sayang dalam keluarga untuk anak disleksia.",
        ],
    },
]

METRICS_CONFIG = [
    {"class": AnswerRelevancyMetric, "threshold": 0.5, "weight": 1.0},
    {"class": HallucinationMetric, "threshold": 0.5, "weight": 1.5},
    {"class": FaithfulnessMetric, "threshold": 0.5, "weight": 1.2},
    {"class": ContextualRelevancyMetric, "threshold": 0.5, "weight": 1.0},
    {"class": ToxicityMetric, "threshold": 0.1, "weight": 2.0},
    {"class": BiasMetric, "threshold": 0.1, "weight": 1.5},
]

def call_llm_ollama(ollama_client, model, prompt):
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DYSLEXIA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def call_llm(client, model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DYSLEXIA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def build_test_cases(client, model):
    test_cases = []

    # ── user simulation (prompts array) ──────────────
    for template in TEMPLATE_TEST_CASES:
        for i, prompt in enumerate(template["prompts"], 1):
            tc_id = f"TC_{template['id_prefix']}_USER_{model}_{i:03d}"
            actual_output = call_llm(client, model, prompt)
            test_cases.append({
                "id": tc_id,
                "category": template["category"],
                "sub_category": template["sub_category"],
                "difficulty_level": template["difficulty_level"],
                "track": "user_simulation",       
                "prompt_strategy": "natural",
                "input": prompt,
                "actual_output": actual_output,
                "expected_output": template["expected_output"],
                "context": template["context"],
                "retrieval_context": template["context"],
            })

    # ── structured_prompts ──────────
    for template in TEMPLATE_TEST_CASES:
        sp = template.get("structured_prompts", {})
        for strategy, prompt in sp.items():
            if not prompt:
                continue
            tc_id = f"TC_{template['id_prefix']}_{strategy.upper()}_{model}"
            actual_output = call_llm(client, model, prompt)
            test_cases.append({
                "id": tc_id,
                "category": template["category"],
                "sub_category": template["sub_category"],
                "difficulty_level": template["difficulty_level"],
                "track": "research",        
                "prompt_strategy": strategy,       # zero_shot / few_shot / cot / hybrid
                "input": prompt,
                "actual_output": actual_output,
                "expected_output": template["expected_output"],
                "context": template["context"],
                "retrieval_context": template["context"],
                
            })

    return test_cases

# def build_test_cases(client, model):
#     test_cases = []
#     case_counter = {}

#     for template in TEMPLATE_TEST_CASES:
#         prefix = template["id_prefix"]
#         case_counter[prefix] = 0
#         for prompt in template["prompts"]:
#             case_counter[prefix] += 1
#             tc_id = f"TC_{prefix}_{model.replace('-', '_').upper()}_{case_counter[prefix]:03d}"

#             actual_output = call_llm(client, model, prompt)
#             time.sleep(0.5)

#             test_cases.append({
#                 "id": tc_id,
#                 "category": template["category"],
#                 "sub_category": template["sub_category"],
#                 "difficulty_level": template["difficulty_level"],
#                 "model": model,
#                 "input": prompt,
#                 "actual_output": actual_output,
#                 "expected_output": template["expected_output"],
#                 "context": template["context"],
                # "retrieval_context": template["context"],
#             })

#     return test_cases


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


def evaluate_test_cases(test_cases_data, weights_map, metrics_instances):
    all_results = []
    raw_scores_by_metric = {cfg["class"].__name__: [] for cfg in METRICS_CONFIG}
    scores_by_category = {}
    scores_by_difficulty = {}
    scores_by_model = {}

    for tc_data in test_cases_data:
        tc_start_time = time.time()

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
            metric_name = metric.__class__.__name__

            metric_start = time.time()
            try:
                metric.measure(test_case)
                score = metric.score if metric.score is not None else 0.0
                reason = metric.reason
                is_successful = metric.is_successful()
            except Exception as e:
                score = 0.0
                reason = f"[METRIC ERROR: {str(e)}]"
                is_successful = False
            metric_elapsed = round(time.time() - metric_start, 4)

            quality_label = classify_score_quality(score, metric_name)
            metric_data = {
                "metric_name": metric_name,
                "score": round(score, 6),
                "threshold": m_info["threshold"],
                "weight": m_info["weight"],
                "is_successful": is_successful,
                "reason": reason,
                "execution_time_seconds": metric_elapsed,
                "quality_classification": quality_label,
            }
            metrics_evaluation.append(metric_data)
            raw_scores_by_metric[metric_name].append(round(score, 6))

            cat = tc_data["category"]
            if cat not in scores_by_category:
                scores_by_category[cat] = {}
            if metric_name not in scores_by_category[cat]:
                scores_by_category[cat][metric_name] = []
            scores_by_category[cat][metric_name].append(round(score, 6))

            diff = tc_data["difficulty_level"]
            if diff not in scores_by_difficulty:
                scores_by_difficulty[diff] = {}
            if metric_name not in scores_by_difficulty[diff]:
                scores_by_difficulty[diff][metric_name] = []
            scores_by_difficulty[diff][metric_name].append(round(score, 6))

            mdl = tc_data["model"]
            if mdl not in scores_by_model:
                scores_by_model[mdl] = {}
            if metric_name not in scores_by_model[mdl]:
                scores_by_model[mdl][metric_name] = []
            scores_by_model[mdl][metric_name].append(round(score, 6))

        tc_elapsed = round(time.time() - tc_start_time, 4)
        pass_count = sum(1 for m in metrics_evaluation if m["is_successful"])
        fail_count = len(metrics_evaluation) - pass_count
        composite_score = compute_composite_score(metrics_evaluation, weights_map)

        tc_result = {
            "test_case_id": tc_data["id"],
            "category": tc_data["category"],
            "sub_category": tc_data["sub_category"],
            "difficulty_level": tc_data["difficulty_level"],
            "model": tc_data["model"],
            "input": tc_data["input"],
            "actual_output": tc_data["actual_output"],
            "expected_output": tc_data["expected_output"],
            "context_provided": tc_data["context"],
            "retrieval_context_provided": tc_data["retrieval_context"],
            "metrics_evaluation": metrics_evaluation,
            "test_case_summary": {
                "total_metrics_evaluated": len(metrics_evaluation),
                "metrics_passed": pass_count,
                "metrics_failed": fail_count,
                "pass_rate": round(pass_count / len(metrics_evaluation), 6) if metrics_evaluation else 0,
                "composite_weighted_score": round(composite_score, 6) if composite_score is not None else None,
                "execution_time_seconds": tc_elapsed,
            },
        }
        all_results.append(tc_result)

    return all_results, raw_scores_by_metric, scores_by_category, scores_by_difficulty, scores_by_model


def compute_aggregate_statistics(raw_scores_by_metric):
    aggregate = {}
    for metric_name, scores in raw_scores_by_metric.items():
        n = len(scores)
        if n == 0:
            continue
        mean_val = safe_mean(scores)
        std_val = safe_std(scores)
        median_val = safe_median(scores)
        min_val = min(scores)
        max_val = max(scores)
        q1 = safe_percentile(scores, 25)
        q3 = safe_percentile(scores, 75)
        iqr = round(q3 - q1, 6) if (q1 is not None and q3 is not None) else None
        cv = round(std_val / mean_val, 6) if (std_val is not None and mean_val and mean_val != 0) else None
        skewness = None
        if n >= 3 and std_val and std_val > 0:
            skewness = round(
                (n / ((n - 1) * (n - 2))) * sum(((x - mean_val) / std_val) ** 3 for x in scores), 6
            )
        kurtosis = None
        if n >= 4 and std_val and std_val > 0:
            kurtosis = round(
                (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)))
                * sum(((x - mean_val) / std_val) ** 4 for x in scores)
                - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3))), 6
            )
        w_stat, p_shapiro = shapiro_wilk_approx(scores)
        ci_lower, ci_upper = confidence_interval_95(scores)
        boot_lower, boot_upper = bootstrap_mean_ci(scores)
        threshold_val = next(
            (cfg["threshold"] for cfg in METRICS_CONFIG if cfg["class"].__name__ == metric_name), 0.5
        )
        inverted = metric_name in {"ToxicityMetric", "HallucinationMetric", "BiasMetric"}
        pass_count = sum(1 for s in scores if (s <= threshold_val if inverted else s >= threshold_val))

        aggregate[metric_name] = {
            "n": n,
            "mean": round(mean_val, 6) if mean_val is not None else None,
            "std": round(std_val, 6) if std_val is not None else None,
            "median": round(median_val, 6) if median_val is not None else None,
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "range": round(max_val - min_val, 6),
            "q1_25th_percentile": round(q1, 6) if q1 is not None else None,
            "q3_75th_percentile": round(q3, 6) if q3 is not None else None,
            "iqr_interquartile_range": iqr,
            "coefficient_of_variation": cv,
            "skewness": skewness,
            "excess_kurtosis": kurtosis,
            "normality_test_shapiro_wilk": {
                "w_statistic": w_stat,
                "p_value_approx": p_shapiro,
                "is_normal_dist_p005": (p_shapiro > 0.05) if p_shapiro is not None else None,
                "note": "Approximation; use scipy.stats.shapiro for exact values.",
            },
            "confidence_interval_95pct": {
                "lower": ci_lower,
                "upper": ci_upper,
                "method": "t-distribution",
            },
            "bootstrap_ci_95pct": {
                "lower": boot_lower,
                "upper": boot_upper,
                "n_bootstrap": 1000,
                "method": "percentile bootstrap",
            },
            "pass_rate": round(pass_count / n, 6) if n > 0 else 0,
            "pass_count": pass_count,
            "fail_count": n - pass_count,
            "all_scores": scores,
        }
    return aggregate


def compute_correlation_matrix(raw_scores_by_metric):
    matrix = {}
    metric_names = list(raw_scores_by_metric.keys())
    for m1, m2 in itertools.combinations(metric_names, 2):
        pearson_r = pearson_correlation(raw_scores_by_metric[m1], raw_scores_by_metric[m2])
        spearman_r = spearman_correlation(raw_scores_by_metric[m1], raw_scores_by_metric[m2])
        n = len(raw_scores_by_metric[m1])
        p_pearson = None
        if pearson_r is not None and n > 2:
            t_stat = pearson_r * math.sqrt(n - 2) / math.sqrt(max(1 - pearson_r ** 2, 1e-10))
            p_pearson = round(2 * (1 - _norm_cdf(abs(t_stat))), 6)
        p_spearman = None
        if spearman_r is not None and n > 2:
            t_stat_s = spearman_r * math.sqrt(n - 2) / math.sqrt(max(1 - spearman_r ** 2, 1e-10))
            p_spearman = round(2 * (1 - _norm_cdf(abs(t_stat_s))), 6)
        key = f"{m1}_vs_{m2}"

        def interp(r):
            if r is None:
                return "undefined"
            if r >= 0.7:
                return "strong_positive"
            if r >= 0.4:
                return "moderate_positive"
            if r > 0:
                return "weak_positive"
            if r <= -0.7:
                return "strong_negative"
            if r <= -0.4:
                return "moderate_negative"
            if r < 0:
                return "weak_negative"
            return "no_correlation"

        matrix[key] = {
            "metric_a": m1,
            "metric_b": m2,
            "pearson_r": round(pearson_r, 6) if pearson_r is not None else None,
            "pearson_p_value": p_pearson,
            "spearman_rho": round(spearman_r, 6) if spearman_r is not None else None,
            "spearman_p_value": p_spearman,
            "interpretation_pearson": interp(pearson_r),
            "interpretation_spearman": interp(spearman_r),
        }
    return matrix


def compute_model_comparison(scores_by_model):
    model_names = list(scores_by_model.keys())
    metric_names = list(METRICS_CONFIG[i]["class"].__name__ for i in range(len(METRICS_CONFIG)))
    comparison = {}

    for metric_name in metric_names:
        model_score_groups = {
            mdl: scores_by_model[mdl].get(metric_name, [])
            for mdl in model_names
        }
        comparison[metric_name] = {}

        for mdl, scores in model_score_groups.items():
            ci_lower, ci_upper = confidence_interval_95(scores)
            boot_lower, boot_upper = bootstrap_mean_ci(scores)
            comparison[metric_name][mdl] = {
                "n": len(scores),
                "mean": round(safe_mean(scores), 6) if safe_mean(scores) is not None else None,
                "std": round(safe_std(scores), 6) if safe_std(scores) is not None else None,
                "median": round(safe_median(scores), 6) if safe_median(scores) is not None else None,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "bootstrap_ci_95_lower": boot_lower,
                "bootstrap_ci_95_upper": boot_upper,
                "scores": scores,
            }

        if len(model_names) >= 2:
            pairwise_mwu = {}
            pairwise_p_values = []
            for m1, m2 in itertools.combinations(model_names, 2):
                g1 = model_score_groups[m1]
                g2 = model_score_groups[m2]
                u_stat, p_val = mann_whitney_u(g1, g2)
                d = cohens_d(g1, g2)
                pairwise_mwu[f"{m1}_vs_{m2}"] = {
                    "mann_whitney_u": u_stat,
                    "p_value": p_val,
                    "cohens_d": round(d, 6) if d is not None else None,
                    "effect_size_classification": classify_effect_size(d),
                }
                if p_val is not None:
                    pairwise_p_values.append(p_val)

            kw_groups = [model_score_groups[mdl] for mdl in model_names if model_score_groups[mdl]]
            h_stat, kw_p = kruskal_wallis(*kw_groups) if len(kw_groups) >= 2 else (None, None)
            bonferroni = bonferroni_correction(pairwise_p_values)

            comparison[metric_name]["statistical_tests"] = {
                "kruskal_wallis": {
                    "h_statistic": h_stat,
                    "p_value": kw_p,
                    "significant_p005": (kw_p < 0.05) if kw_p is not None else None,
                },
                "pairwise_mann_whitney_u": pairwise_mwu,
                "bonferroni_correction": bonferroni,
            }

    return comparison


def compute_category_performance(scores_by_category):
    result = {}
    for cat, metric_scores in scores_by_category.items():
        result[cat] = {}
        for metric_name, scores in metric_scores.items():
            ci_lower, ci_upper = confidence_interval_95(scores)
            result[cat][metric_name] = {
                "n": len(scores),
                "mean": round(safe_mean(scores), 6) if safe_mean(scores) is not None else None,
                "std": round(safe_std(scores), 6) if safe_std(scores) is not None else None,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "scores": scores,
            }
    return result


def compute_difficulty_performance(scores_by_difficulty):
    result = {}
    for diff, metric_scores in scores_by_difficulty.items():
        result[diff] = {}
        for metric_name, scores in metric_scores.items():
            ci_lower, ci_upper = confidence_interval_95(scores)
            result[diff][metric_name] = {
                "n": len(scores),
                "mean": round(safe_mean(scores), 6) if safe_mean(scores) is not None else None,
                "std": round(safe_std(scores), 6) if safe_std(scores) is not None else None,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "scores": scores,
            }
    return result


def main():
    experiment_start_time = time.time()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ollama_client = Client(host='https://ollama.com',headers={'Authorization': 'Bearer ' + os.environ["OLLAMA_API_KEY"]})
    weights_map = {cfg["class"].__name__: cfg["weight"] for cfg in METRICS_CONFIG}

    metrics_instances = [
        {
            "instance": cfg["class"](threshold=cfg["threshold"]),
            "threshold": cfg["threshold"],
            "weight": cfg["weight"],
        }
        for cfg in METRICS_CONFIG
    ]

    all_test_cases_data = []
    for model in MODELS_TO_EVALUATE:
        print(f"\n[INFO] Generating test cases untuk model: {model}")
        model_test_cases = build_test_cases(client, model)
        all_test_cases_data.extend(model_test_cases)
        print(f"[INFO] Total test cases untuk {model}: {len(model_test_cases)}")

    print(f"\n[INFO] Total seluruh test cases: {len(all_test_cases_data)}")
    print(f"[INFO] Mulai evaluasi metrik DeepEval...")

    (
        all_test_case_results,
        raw_scores_by_metric,
        scores_by_category,
        scores_by_difficulty,
        scores_by_model,
    ) = evaluate_test_cases(all_test_cases_data, weights_map, metrics_instances)

    aggregate_statistics = compute_aggregate_statistics(raw_scores_by_metric)
    metric_correlation_matrix = compute_correlation_matrix(raw_scores_by_metric)
    model_comparison = compute_model_comparison(scores_by_model)
    performance_by_category = compute_category_performance(scores_by_category)
    performance_by_difficulty = compute_difficulty_performance(scores_by_difficulty)

    all_composite_scores = [
        tc["test_case_summary"]["composite_weighted_score"]
        for tc in all_test_case_results
        if tc["test_case_summary"]["composite_weighted_score"] is not None
    ]
    all_pass_rates = [tc["test_case_summary"]["pass_rate"] for tc in all_test_case_results]
    all_exec_times = [tc["test_case_summary"]["execution_time_seconds"] for tc in all_test_case_results]

    total_metrics_passed = sum(tc["test_case_summary"]["metrics_passed"] for tc in all_test_case_results)
    total_metrics_evaluated = sum(tc["test_case_summary"]["total_metrics_evaluated"] for tc in all_test_case_results)
    experiment_total_time = round(time.time() - experiment_start_time, 4)

    raw_scores_matrix = {
        tc["test_case_id"]: {m["metric_name"]: m["score"] for m in tc["metrics_evaluation"]}
        for tc in all_test_case_results
    }

    comp_ci_lower, comp_ci_upper = confidence_interval_95(all_composite_scores)
    comp_boot_lower, comp_boot_upper = bootstrap_mean_ci(all_composite_scores)

    output_document = {
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
        "test_cases": all_test_case_results,
        "aggregate_statistics": aggregate_statistics,
        "metric_correlation_matrix": metric_correlation_matrix,
        "model_comparison_analysis": model_comparison,
        "performance_by_category": performance_by_category,
        "performance_by_difficulty_level": performance_by_difficulty,
        "overall_experiment_summary": {
            "total_test_cases": len(all_test_case_results),
            "total_metrics_evaluated": total_metrics_evaluated,
            "total_metrics_passed": total_metrics_passed,
            "total_metrics_failed": total_metrics_evaluated - total_metrics_passed,
            "overall_pass_rate": round(total_metrics_passed / total_metrics_evaluated, 6) if total_metrics_evaluated > 0 else 0,
            "composite_score_statistics": {
                "mean": round(safe_mean(all_composite_scores), 6) if safe_mean(all_composite_scores) is not None else None,
                "std": round(safe_std(all_composite_scores), 6) if safe_std(all_composite_scores) is not None else None,
                "median": round(safe_median(all_composite_scores), 6) if safe_median(all_composite_scores) is not None else None,
                "min": round(min(all_composite_scores), 6) if all_composite_scores else None,
                "max": round(max(all_composite_scores), 6) if all_composite_scores else None,
                "ci_95_lower": comp_ci_lower,
                "ci_95_upper": comp_ci_upper,
                "bootstrap_ci_95_lower": comp_boot_lower,
                "bootstrap_ci_95_upper": comp_boot_upper,
            },
            "pass_rate_statistics": {
                "mean": round(safe_mean(all_pass_rates), 6) if safe_mean(all_pass_rates) is not None else None,
                "std": round(safe_std(all_pass_rates), 6) if safe_std(all_pass_rates) is not None else None,
                "min": round(min(all_pass_rates), 6) if all_pass_rates else None,
                "max": round(max(all_pass_rates), 6) if all_pass_rates else None,
            },
            "execution_time_statistics": {
                "total_seconds": experiment_total_time,
                "mean_per_test_case_seconds": round(safe_mean(all_exec_times), 4) if safe_mean(all_exec_times) is not None else None,
                "min_seconds": round(min(all_exec_times), 4) if all_exec_times else None,
                "max_seconds": round(max(all_exec_times), 4) if all_exec_times else None,
            },
        },
        "raw_scores_matrix": raw_scores_matrix,
    }

    output_path = "evaluasi_llm_disleksia_scopus_q1_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_document, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluasi selesai. Hasil disimpan di: {output_path}")
    print(f"Total test cases: {len(all_test_case_results)}")
    print(f"Total model dievaluasi: {len(MODELS_TO_EVALUATE)}")
    print(f"Total evaluasi metrik: {total_metrics_evaluated}")
    print(
        f"Overall pass rate: "
        f"{round(total_metrics_passed / total_metrics_evaluated * 100, 2) if total_metrics_evaluated else 0}%"
    )
    print(f"Total waktu eksekusi: {experiment_total_time} detik")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
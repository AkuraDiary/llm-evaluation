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
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY not set")


MODELS_TO_EVALUATE = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

MODELS_TO_EVALUATE_OLLAMA = [
    "gemma4:31b-cloud",
    "gpt-oss:20b-cloud"
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
import json
import os
import time
import math
import platform
import sys
import itertools
import random
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── API / connection settings ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-oss:120b-cloud") #deepseek-v3.1:671b-cloud

# Ollama models
MODELS_TO_EVALUATE_OLLAMA = [
    "ministral-3:14b-cloud",
    "gpt-oss:20b-cloud",
    "deepseek-v3.2:cloud",
    "gemma4:31b-cloud",
    "deepseek-v4-flash:cloud",
]

# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_DYSLEXIA = (
    "Kamu adalah guru sekolah dasar yang ahli dalam membuat soal untuk anak-anak dengan disleksia. "
    "Ikuti aturan berikut dengan ketat:\n"
    "1. Gunakan kalimat sangat pendek, maksimal 8 kata per kalimat.\n"
    "2. Gunakan kata-kata sederhana sehari-hari, hindari istilah teknis.\n"
    "3. Gunakan objek konkret yang mudah divisualisasikan.\n"
    "4. Angka harus kecil, antara 1 sampai 20.\n"
    "5. Sertakan jawaban secara eksplisit di akhir.\n"
    "6. Jangan gunakan kalimat majemuk bertingkat.\n"
    "Langsung berikan pasangan soal dan jawaban tanpa penjelasan tambahan."
)

# ── Test-case templates ──────────────────────────────────────────────────────
"""
TEMPLATE_TEST_CASES — filled structured_prompts
================================================
Strategy design rationale (grounded in your journals):

ZERO_SHOT  : Bare task instruction only. System prompt does all the work.
             Control condition. (Al Nazi et al., 2025 — baseline)

FEW_SHOT   : 3 complete input-output example pairs before the task.
             3-shot is the empirical sweet spot (Al Nazi Table 5: 94.87% accuracy).
             Examples themselves are dyslexia-compliant → model mirrors the style.

COT        : 3 explicit reasoning steps embedded in the user prompt.
             Capped at 3 steps — longer chains cause hallucinations on smaller models
             (Al Nazi: GPT-3.5 CoT sentiment collapsed from 37% → 2%).
             Steps constrain the model toward dyslexia compliance rules.

HYBRID     : 2 condensed examples + 3-step CoT in one prompt.
             Combines few-shot style priming with explicit reasoning scaffolding.
             Based on Kim et al. (2023) CoT Collection: example + reasoning outperforms either alone.
"""

from datetime import datetime, timezone

TEMPLATE_TEST_CASES = [
    # ── 1. MATEMATIKA — PENJUMLAHAN ──────────────────────────────────────────
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
        "structured_prompts": {
            # ── ZERO-SHOT ────────────────────────────────────────────────────
            "zero_shot": (
                "Buat 1 soal penjumlahan untuk anak disleksia bertema apel. "
                "Sertakan jawaban."
            ),

            # ── FEW-SHOT (3 examples) ────────────────────────────────────────
            "few_shot": (
                "Contoh 1:\n"
                "Soal: Ada 3 jeruk. Ibu beli 2 lagi. Berapa jeruknya?\n"
                "Jawaban: 3 + 2 = 5. Jeruknya ada 5.\n\n"

                "Contoh 2:\n"
                "Soal: Ada 4 buku. Kamu dapat 1 lagi. Berapa bukunya?\n"
                "Jawaban: 4 + 1 = 5. Bukunya ada 5.\n\n"

                "Contoh 3:\n"
                "Soal: Ada 2 kucing. Datang 3 lagi. Berapa kucingnya?\n"
                "Jawaban: 2 + 3 = 5. Kucingnya ada 5.\n\n"

                "Sekarang buat 1 soal penjumlahan bertema apel. "
                "Ikuti format contoh di atas."
            ),

            # ── COT (3 steps) ─────────────────────────────────────────────────
            "cot": (
                "Buat 1 soal penjumlahan bertema apel untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih 2 angka kecil antara 1 sampai 10.\n"
                "Langkah 2: Tulis soal maksimal 8 kata menggunakan kata 'apel'.\n"
                "Langkah 3: Tulis jawaban dengan persamaan (contoh: 3 + 4 = 7).\n\n"
                "Mulai sekarang."
            ),

            # ── HYBRID (2 examples + CoT steps) ──────────────────────────────
            "hybrid": (
                "Contoh cara membuat soal:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih angka: 3 dan 2\n"
                "Langkah 2 → tulis soal: 'Ada 3 jeruk. Ibu beli 2 lagi. Berapa?'\n"
                "Langkah 3 → tulis jawaban: '3 + 2 = 5. Jeruknya ada 5.'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih angka: 5 dan 3\n"
                "Langkah 2 → tulis soal: 'Ada 5 bola. Dapat 3 lagi. Berapa?'\n"
                "Langkah 3 → tulis jawaban: '5 + 3 = 8. Bolanya ada 8.'\n\n"
                "Sekarang buat 1 soal penjumlahan bertema apel. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 2. MATEMATIKA — PENGURANGAN ──────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pengurangan untuk anak disleksia bertema jeruk. "
                "Sertakan jawaban."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Ada 8 apel. Dimakan 3. Berapa sisanya?\n"
                "Jawaban: 8 - 3 = 5. Sisanya 5 apel.\n\n"

                "Contoh 2:\n"
                "Soal: Ada 7 bola. Hilang 2. Berapa sisanya?\n"
                "Jawaban: 7 - 2 = 5. Sisanya 5 bola.\n\n"

                "Contoh 3:\n"
                "Soal: Ada 9 ikan. Dijual 4. Berapa sisanya?\n"
                "Jawaban: 9 - 4 = 5. Sisanya 5 ikan.\n\n"

                "Sekarang buat 1 soal pengurangan bertema jeruk. "
                "Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pengurangan bertema jeruk untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih angka awal (5-10) dan angka yang dikurangi (1-4).\n"
                "Langkah 2: Tulis soal maksimal 8 kata menggunakan kata 'jeruk'.\n"
                "Langkah 3: Tulis jawaban dengan persamaan pengurangan yang lengkap.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal pengurangan:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih angka: 8 dan 3\n"
                "Langkah 2 → tulis soal: 'Ada 8 apel. Dimakan 3. Berapa sisanya?'\n"
                "Langkah 3 → tulis jawaban: '8 - 3 = 5. Sisanya 5 apel.'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih angka: 6 dan 2\n"
                "Langkah 2 → tulis soal: 'Ada 6 kue. Dimakan 2. Berapa sisanya?'\n"
                "Langkah 3 → tulis jawaban: '6 - 2 = 4. Sisanya 4 kue.'\n\n"
                "Sekarang buat 1 soal pengurangan bertema jeruk. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 3. MATEMATIKA — PERKALIAN ─────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal perkalian untuk anak disleksia bertema kantong permen. "
                "Sertakan petunjuk cara menghitung dan jawaban."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Ada 2 piring. Tiap piring ada 3 kue. Berapa semua kuenya?\n"
                "Hitung: 2 x 3 = ?\n"
                "Jawaban: 2 x 3 = 6. Semua kuenya ada 6.\n\n"

                "Contoh 2:\n"
                "Soal: Ada 3 kotak. Tiap kotak ada 4 pensil. Berapa semua pensilnya?\n"
                "Hitung: 3 x 4 = ?\n"
                "Jawaban: 3 x 4 = 12. Semua pensilnya ada 12.\n\n"

                "Contoh 3:\n"
                "Soal: Ada 4 vas. Tiap vas ada 2 bunga. Berapa semua bunganya?\n"
                "Hitung: 4 x 2 = ?\n"
                "Jawaban: 4 x 2 = 8. Semua bunganya ada 8.\n\n"

                "Sekarang buat 1 soal perkalian bertema kantong permen. "
                "Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal perkalian bertema kantong permen untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Tentukan jumlah kantong (2-4) dan isi tiap kantong (2-5).\n"
                "Langkah 2: Tulis soal pendek dengan kata 'kantong' dan 'permen'.\n"
                "Langkah 3: Tulis petunjuk 'Hitung: A x B = ?' lalu tulis jawaban lengkap.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal perkalian:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → tentukan: 2 piring, tiap piring 3 kue\n"
                "Langkah 2 → tulis soal: 'Ada 2 piring. Tiap piring ada 3 kue. Berapa?'\n"
                "Langkah 3 → tulis jawaban: 'Hitung: 2 x 3 = 6. Semua kuenya ada 6.'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → tentukan: 3 kotak, tiap kotak 4 pensil\n"
                "Langkah 2 → tulis soal: 'Ada 3 kotak. Tiap kotak ada 4 pensil. Berapa?'\n"
                "Langkah 3 → tulis jawaban: 'Hitung: 3 x 4 = 12. Semua pensilnya ada 12.'\n\n"
                "Sekarang buat 1 soal perkalian bertema kantong permen. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 4. MATEMATIKA — PEMBAGIAN ─────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pembagian untuk anak disleksia bertema cokelat. "
                "Sertakan jawaban."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Ada 8 permen. Dibagi 2 anak. Tiap anak dapat berapa?\n"
                "Jawaban: 8 : 2 = 4. Tiap anak dapat 4 permen.\n\n"

                "Contoh 2:\n"
                "Soal: Ada 6 apel. Dibagi 3 anak. Tiap anak dapat berapa?\n"
                "Jawaban: 6 : 3 = 2. Tiap anak dapat 2 apel.\n\n"

                "Contoh 3:\n"
                "Soal: Ada 9 roti. Dibagi 3 anak. Tiap anak dapat berapa?\n"
                "Jawaban: 9 : 3 = 3. Tiap anak dapat 3 roti.\n\n"

                "Sekarang buat 1 soal pembagian bertema cokelat. "
                "Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pembagian bertema cokelat untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih jumlah cokelat (6-10) yang habis dibagi rata.\n"
                "Langkah 2: Tulis soal pendek, maksimal 8 kata.\n"
                "Langkah 3: Tulis jawaban dengan persamaan pembagian (contoh: 6 : 2 = 3).\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal pembagian:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih angka: 8 cokelat dibagi 2 anak\n"
                "Langkah 2 → tulis soal: 'Ada 8 cokelat. Dibagi 2 anak. Berapa tiap anak?'\n"
                "Langkah 3 → tulis jawaban: '8 : 2 = 4. Tiap anak dapat 4 cokelat.'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih angka: 6 kelereng dibagi 3 anak\n"
                "Langkah 2 → tulis soal: 'Ada 6 kelereng. Dibagi 3 anak. Berapa tiap anak?'\n"
                "Langkah 3 → tulis jawaban: '6 : 3 = 2. Tiap anak dapat 2 kelereng.'\n\n"
                "Sekarang buat 1 soal pembagian bertema cokelat. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 5. BAHASA INDONESIA — MEMBACA ─────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat teks bacaan pendek dan 1 soal pemahaman "
                "untuk anak disleksia bertema hewan peliharaan. "
                "Sertakan jawaban."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Teks: Budi punya kucing. Kucingnya berwarna putih. Ia sayang sekali.\n"
                "Soal: Apa warna kucing Budi?\n"
                "Jawaban: Warna kucing Budi adalah putih.\n\n"

                "Contoh 2:\n"
                "Teks: Ani makan nasi. Nasinya hangat. Ani merasa kenyang.\n"
                "Soal: Bagaimana rasa nasi Ani?\n"
                "Jawaban: Nasi Ani terasa hangat.\n\n"

                "Contoh 3:\n"
                "Teks: Dito pergi ke sekolah. Ia naik sepeda. Sekolahnya dekat rumah.\n"
                "Soal: Dito naik apa ke sekolah?\n"
                "Jawaban: Dito naik sepeda ke sekolah.\n\n"

                "Sekarang buat teks 2-3 kalimat bertema hewan peliharaan "
                "dan 1 soal pemahaman. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat teks bacaan dan 1 soal pemahaman "
                "bertema hewan peliharaan untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Tulis teks 2-3 kalimat. Tiap kalimat maksimal 6 kata.\n"
                "Langkah 2: Buat 1 pertanyaan tentang fakta yang ada dalam teks.\n"
                "Langkah 3: Tulis jawaban yang bisa ditemukan langsung di teks.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal membaca:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → tulis teks: 'Budi punya kucing. Kucingnya putih. Ia sayang sekali.'\n"
                "Langkah 2 → buat soal: 'Apa warna kucing Budi?'\n"
                "Langkah 3 → tulis jawaban: 'Warna kucing Budi adalah putih.'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → tulis teks: 'Ani punya kelinci. Kelincinya berbulu lebat. Ia senang sekali.'\n"
                "Langkah 2 → buat soal: 'Hewan apa yang dimiliki Ani?'\n"
                "Langkah 3 → tulis jawaban: 'Ani punya kelinci.'\n\n"
                "Sekarang buat teks bertema hewan peliharaan dan 1 soal. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 6. BAHASA INDONESIA — MENULIS ─────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal kalimat rumpang untuk anak disleksia "
                "bertema aktivitas sehari-hari. "
                "Sediakan 3 pilihan kata dan jawaban."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Ibu memasak di ___. (dapur / taman / sekolah)\n"
                "Jawaban: dapur\n\n"

                "Contoh 2:\n"
                "Soal: Aku tidur di atas ___. (kasur / meja / kursi)\n"
                "Jawaban: kasur\n\n"

                "Contoh 3:\n"
                "Soal: Aku minum ___ saat haus. (air / buku / pensil)\n"
                "Jawaban: air\n\n"

                "Sekarang buat 1 soal kalimat rumpang baru "
                "bertema aktivitas sehari-hari. "
                "Sediakan 3 pilihan kata. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal kalimat rumpang untuk anak disleksia "
                "bertema aktivitas sehari-hari.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Tulis kalimat pendek (maks 7 kata) dengan satu kata dikosongkan (___).\n"
                "Langkah 2: Buat 3 pilihan kata yang sangat berbeda maknanya.\n"
                "Langkah 3: Tulis jawaban yang benar.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal kalimat rumpang:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → tulis kalimat: 'Ibu memasak di ___.'\n"
                "Langkah 2 → buat pilihan: '(dapur / taman / sekolah)'\n"
                "Langkah 3 → tulis jawaban: 'dapur'\n\n"
                "Contoh 2:\n"
                "Langkah 1 → tulis kalimat: 'Aku tidur di atas ___.'\n"
                "Langkah 2 → buat pilihan: '(kasur / meja / kursi)'\n"
                "Langkah 3 → tulis jawaban: 'kasur'\n\n"
                "Sekarang buat 1 soal kalimat rumpang baru "
                "bertema aktivitas sehari-hari. Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 7. IPA — HEWAN ────────────────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pilihan ganda IPA tentang makanan hewan "
                "untuk anak disleksia. "
                "Sertakan 3 pilihan jawaban yang kontras dan jawaban yang benar."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Kucing makan apa?\n"
                "A. Ikan   B. Rumput   C. Batu\n"
                "Jawaban: A. Ikan\n\n"

                "Contoh 2:\n"
                "Soal: Sapi makan apa?\n"
                "A. Daging   B. Rumput   C. Ikan\n"
                "Jawaban: B. Rumput\n\n"

                "Contoh 3:\n"
                "Soal: Ayam makan apa?\n"
                "A. Batu   B. Susu   C. Biji-bijian\n"
                "Jawaban: C. Biji-bijian\n\n"

                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang makanan hewan lain. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pilihan ganda IPA tentang makanan hewan "
                "untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih 1 hewan yang dikenal anak sehari-hari.\n"
                "Langkah 2: Tulis pertanyaan maksimal 5 kata.\n"
                "Langkah 3: Buat 3 pilihan (A, B, C) yang sangat berbeda, "
                "lalu tandai jawaban yang benar.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal pilihan ganda hewan:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih hewan: kucing\n"
                "Langkah 2 → tulis soal: 'Kucing makan apa?'\n"
                "Langkah 3 → buat pilihan: 'A. Ikan  B. Rumput  C. Batu' → Jawaban: A\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih hewan: sapi\n"
                "Langkah 2 → tulis soal: 'Sapi makan apa?'\n"
                "Langkah 3 → buat pilihan: 'A. Daging  B. Rumput  C. Ikan' → Jawaban: B\n\n"
                "Sekarang buat 1 soal pilihan ganda tentang makanan hewan lain. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 8. IPA — TUMBUHAN ─────────────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pilihan ganda IPA tentang bagian tumbuhan "
                "untuk anak disleksia. "
                "Sertakan 3 pilihan jawaban yang kontras dan jawaban yang benar."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Daun tumbuhan biasanya berwarna apa?\n"
                "A. Hijau   B. Merah   C. Biru\n"
                "Jawaban: A. Hijau\n\n"

                "Contoh 2:\n"
                "Soal: Tumbuhan minum dari mana?\n"
                "A. Udara   B. Tanah   C. Batu\n"
                "Jawaban: B. Tanah\n\n"

                "Contoh 3:\n"
                "Soal: Bagian tumbuhan yang ada di bawah tanah?\n"
                "A. Bunga   B. Daun   C. Akar\n"
                "Jawaban: C. Akar\n\n"

                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang bagian tumbuhan. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pilihan ganda IPA tentang bagian tumbuhan "
                "untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih 1 bagian tumbuhan yang mudah dilihat anak "
                "(daun, akar, bunga, buah, batang).\n"
                "Langkah 2: Tulis pertanyaan pendek maksimal 8 kata. "
                "Jangan gunakan istilah ilmiah.\n"
                "Langkah 3: Buat 3 pilihan jawaban yang sangat berbeda, "
                "lalu tandai jawaban yang benar.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal pilihan ganda tumbuhan:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih bagian: daun\n"
                "Langkah 2 → tulis soal: 'Daun tumbuhan biasanya berwarna apa?'\n"
                "Langkah 3 → buat pilihan: 'A. Hijau  B. Merah  C. Biru' → Jawaban: A\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih bagian: akar\n"
                "Langkah 2 → tulis soal: 'Bagian tumbuhan yang ada di bawah tanah?'\n"
                "Langkah 3 → buat pilihan: 'A. Bunga  B. Daun  C. Akar' → Jawaban: C\n\n"
                "Sekarang buat 1 soal pilihan ganda tentang bagian tumbuhan lain. "
                "Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 9. IPS — LINGKUNGAN ───────────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pilihan ganda IPS tentang tempat di sekitar rumah "
                "untuk anak disleksia. "
                "Sertakan 3 pilihan jawaban yang kontras dan jawaban yang benar."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Di mana kita beli sayur?\n"
                "A. Pasar   B. Sekolah   C. Rumah Sakit\n"
                "Jawaban: A. Pasar\n\n"

                "Contoh 2:\n"
                "Soal: Di mana kita belajar?\n"
                "A. Pasar   B. Sekolah   C. Sawah\n"
                "Jawaban: B. Sekolah\n\n"

                "Contoh 3:\n"
                "Soal: Di mana dokter bekerja?\n"
                "A. Kebun   B. Pasar   C. Rumah Sakit\n"
                "Jawaban: C. Rumah Sakit\n\n"

                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang tempat di sekitar rumah. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pilihan ganda IPS tentang tempat di sekitar rumah "
                "untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih 1 tempat yang dikenal anak (pasar, sekolah, masjid, dll).\n"
                "Langkah 2: Tulis pertanyaan pendek tentang fungsi tempat itu.\n"
                "Langkah 3: Buat 3 pilihan tempat yang sangat berbeda fungsinya, "
                "lalu tandai jawaban yang benar.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal tempat di sekitar rumah:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih tempat: pasar\n"
                "Langkah 2 → tulis soal: 'Di mana kita beli sayur?'\n"
                "Langkah 3 → buat pilihan: 'A. Pasar  B. Sekolah  C. Rumah Sakit' → Jawaban: A\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih tempat: sekolah\n"
                "Langkah 2 → tulis soal: 'Di mana kita belajar?'\n"
                "Langkah 3 → buat pilihan: 'A. Pasar  B. Sekolah  C. Sawah' → Jawaban: B\n\n"
                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang tempat di sekitar rumah. Ikuti 3 langkah di atas."
            ),
        },
    },

    # ── 10. IPS — KELUARGA ────────────────────────────────────────────────────
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
        "structured_prompts": {
            "zero_shot": (
                "Buat 1 soal pilihan ganda IPS tentang peran anggota keluarga "
                "untuk anak disleksia. "
                "Sertakan 3 pilihan jawaban yang kontras dan jawaban yang benar."
            ),

            "few_shot": (
                "Contoh 1:\n"
                "Soal: Siapa yang memasak di rumah?\n"
                "A. Ibu   B. Dokter   C. Guru\n"
                "Jawaban: A. Ibu\n\n"

                "Contoh 2:\n"
                "Soal: Siapa yang mengajar di sekolah?\n"
                "A. Ayah   B. Ibu   C. Guru\n"
                "Jawaban: C. Guru\n\n"

                "Contoh 3:\n"
                "Soal: Siapa yang mencari nafkah di keluarga?\n"
                "A. Ayah   B. Adik   C. Kucing\n"
                "Jawaban: A. Ayah\n\n"

                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang peran anggota keluarga. Ikuti format contoh di atas."
            ),

            "cot": (
                "Buat 1 soal pilihan ganda IPS tentang peran anggota keluarga "
                "untuk anak disleksia.\n\n"
                "Ikuti 3 langkah ini:\n"
                "Langkah 1: Pilih 1 kegiatan yang dilakukan anggota keluarga di rumah.\n"
                "Langkah 2: Tulis pertanyaan pendek maksimal 6 kata.\n"
                "Langkah 3: Buat 3 pilihan yang sangat berbeda "
                "(1 benar, 2 jelas salah), lalu tandai jawaban yang benar.\n\n"
                "Mulai sekarang."
            ),

            "hybrid": (
                "Contoh cara membuat soal peran keluarga:\n\n"
                "Contoh 1:\n"
                "Langkah 1 → pilih kegiatan: memasak\n"
                "Langkah 2 → tulis soal: 'Siapa yang memasak di rumah?'\n"
                "Langkah 3 → buat pilihan: 'A. Ibu  B. Dokter  C. Guru' → Jawaban: A\n\n"
                "Contoh 2:\n"
                "Langkah 1 → pilih kegiatan: mencari nafkah\n"
                "Langkah 2 → tulis soal: 'Siapa yang mencari nafkah?'\n"
                "Langkah 3 → buat pilihan: 'A. Ayah  B. Adik  C. Kucing' → Jawaban: A\n\n"
                "Sekarang buat 1 soal pilihan ganda baru "
                "tentang peran anggota keluarga. Ikuti 3 langkah di atas."
            ),
        },
    },
]



# ── Word lists ────────────────────────────────────────────────────────────────
TOXIC_WORDS_ID = [
    "bodoh", "goblok", "idiot", "tolol", "dungu", "bangsat", "anjing",
    "babi", "sial", "bajingan", "keparat", "asu", "kampret", "brengsek",
]

BIAS_WORDS_ID = [
    "perempuan tidak bisa", "laki-laki lebih", "wanita lemah", "pria superior",
    "anak miskin", "orang kaya saja", "hanya untuk pintar", "si bodoh",
]

COMPLEX_WORDS_ID = [
    "mengkalkulasi", "mendistribusikan", "ekuivalen", "proporsi", "inferensikan",
    "determinasi", "antusias", "autotrof", "reduksi-oksidasi", "intermediat",
    "komprehensif", "mekanisme", "molekuler", "ekosistem", "biokimia",
    "prasejarah", "peradaban", "perpustakaan", "terpencil", "menjulang",
]

# ── Metric settings ──────────────────────────────────────────────────────────
METRIC_NAMES = [
    "answer_relevancy",
    "context_relevancy",
    "faithfulness",
    "toxicity",
    "bias",
    "dyslexia_compliance",
    "llm_judge_score",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "bleu_avg",
    "flesch_reading_ease",
    "avg_sentence_length_words",
    "type_token_ratio",
]

METRIC_WEIGHTS = {
    "answer_relevancy": 1.0,
    "context_relevancy": 1.0,
    "faithfulness": 1.2,
    "toxicity": 2.0,
    "bias": 1.5,
    "dyslexia_compliance": 2.5,
    "llm_judge_score": 2.0,
    "rouge1_f1": 0.8,
    "rouge2_f1": 0.6,
    "rougeL_f1": 0.8,
    "bleu_avg": 0.6,
    "flesch_reading_ease": 1.5,
    "avg_sentence_length_words": 1.0,
    "type_token_ratio": 0.5,
}

INVERTED_METRICS = {"toxicity", "bias", "avg_sentence_length_words"}

# ── Experiment metadata ───────────────────────────────────────────────────────
EXPERIMENT_START_UTC = datetime.now(timezone.utc).isoformat()
# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for tc in TEMPLATE_TEST_CASES:
        sp = tc.get("structured_prompts", {})
        missing = [k for k, v in sp.items() if not v]
        status = "✓" if not missing else f"✗ missing: {missing}"
        print(f"{tc['id_prefix']:25s} {status}")
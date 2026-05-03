import json
import re


def tokenize(text):
    """Lowercase, strip punctuation, and split into word tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if t]


def json_exporter(output_data_json, output_file):
    output_path = "outputs/" + output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data_json, f, indent=4, ensure_ascii=False)
    print(f"Hasil disimpan di: {output_path}")

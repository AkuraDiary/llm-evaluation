import json
def json_exporter(output_data_json, output_file):
    output_path = "output/" + output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data_json, f, indent=4, ensure_ascii=False)

    print(f"Hasil disimpan di: {output_path}")

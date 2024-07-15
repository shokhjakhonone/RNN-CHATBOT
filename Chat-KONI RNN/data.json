import json

input_json_path = "intents.json"  

output_json_path = "output.json"  # Замените путем к выходному JSON-файлу

with open(input_json_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)


examples = [
    {"text": f"<s>[INST] {intent['question']} [/INST] {intent['answer']} </s>"}
    for intent in data["intents"]
]

# Сохранение в новый JSON-файл
with open(output_json_path, "w", encoding="utf-8") as output_json_file:
    json.dump({"train": examples}, output_json_file, ensure_ascii=False, indent=2)

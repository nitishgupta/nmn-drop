import json

input_json = "/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno_final.json"

output_json = "/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno_final2.json"

with open(input_json, 'r') as f:
    dataset = json.load(f)


with open(output_json, 'w') as f:
    json.dump(dataset, f, indent=4)



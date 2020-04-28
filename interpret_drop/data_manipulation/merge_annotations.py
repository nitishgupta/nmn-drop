import json

def read_data(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def merge_data(data_1, data_2):
    print("Sizes: {} and {}".format(len(data_1), len(data_2)))

    merged_data = {}
    merged_data.update(data_1)
    merged_data.update(data_2)

    return merged_data


def write_data(data, output_json):
    with open(output_json, 'w') as f:
        json.dump(data, f)
    print(output_json)
    print("Done!")


def main(annotations_json_1, annotations_json_2, output_json):
    data_1 = read_data(annotations_json_1)
    data_2 = read_data(annotations_json_2)
    merged_data = merge_data(data_1, data_2)

    write_data(merged_data, output_json)




if __name__=="__main__":
    annotations_json_1 = "/shared/nitishg/data/interpret-drop/v2/interpret_dev/interpret_dev_manual_wanno.json"
    annotations_json_2 = "/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno.json"
    output_json = "/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno_v2.json"
    main(annotations_json_1, annotations_json_2, output_json)

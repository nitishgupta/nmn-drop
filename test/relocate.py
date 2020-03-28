from typing import List, Dict, Optional
import json

class Output:
    def __init__(self, input_name: str, values: List[float], label: Optional[str] = None):
        self.input_name = input_name
        self.values = values
        self.label = label

    def toJSON(self):
        json_dict = {
            "input_name": self.input_name,
            "values": self.values,
            "label": self.label
        }
        return json_dict


o1 = Output(input_name="passage", values=[1, 2])
o2 = Output(input_name="numbers", values=[1, 2], label="n1")
outputs = [o1, o2]

outputs_json = [o1.toJSON(), o2.toJSON()]

s = json.dumps({"outputs": outputs_json})

print(s)




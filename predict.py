from PIL import Image
import json
from os import listdir
import numpy as np
import requests


def load_parameters(config_file_path):
    with open(config_file_path, 'r') as file:
        data = json.load(file)
        image_path = data["image_dir"]
        scoring_uri = data["scoring_uri"]
        return image_path, scoring_uri


def load_images_in_batches(image_path, image_names, batch_size):
    """Load images and convert them to json compatible nested lists"""
    batch = []
    for i, image_name in enumerate(image_names):
        image = Image.open(image_path+image_name)
        np_image = np.array(image)
        image_list = np_image.tolist()
        batch.append(image_list)

    print("List Shape (in terms of nesting):", (len(batch),
                        len(batch[0]), len(batch[0][0]), 
                        len(batch[0][0][0])))
    return batch


def get_images(): # Does not support multiple batches! Only 1 batch for testing purposes.
    image_names = listdir(image_path)
    batch_size = len(image_names)
    return image_path, image_names, batch_size



image_path, scoring_uri = load_parameters("config.json")
image_path, image_names, batch_size = get_images()
image_batch = load_images_in_batches(image_path, image_names, batch_size)


input_data_json = json.dumps({"data": [image_batch]})
headers = {"Content-Type": "application/json"}
response = requests.post(scoring_uri, data=input_data_json, headers=headers)

if response.status_code == 200:
    result = json.loads(response.json())
    print(result)
    prediction = result["result"][0]
    print(f"Prediction: {prediction}")
else:
    print(f"Error: {response.text}")

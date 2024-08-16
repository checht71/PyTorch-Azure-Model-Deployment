import json
import numpy as np
from azureml.core.model import Model
import torch
from torchvision.transforms import v2
from PIL import Image
import torch.nn as nn
import torch
import torchvision.models as models

def init():
    global model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    config_file_path = "config.json"

    with open(config_file_path, 'r') as file:
        weights_path = json.load(file)
        print(weights_path["weights_path"])

    state_dict = torch.load(weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print('Model loaded and set to evaluation mode')
    if torch.cuda.is_available():
        print('CUDA is available and model is on GPU')
    else:
        print('CUDA is not available, model is on CPU')


def pil_images_to_tensor(pil_images):
    transforms = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(128, 128)), v2.Normalize((0.5,), (0.5,))])
    tensors = [transforms(img) for img in pil_images]
    image_tensor = torch.stack(tensors)
    return image_tensor


def run(raw_data):
    data = json.loads(raw_data)['data']
    np_array = np.array(data).squeeze(0)
    pil_image = [Image.fromarray(np_image.astype(np.uint8)) for np_image in np_array]
    input_data = pil_images_to_tensor(pil_image)
    output_tensor = torch.argmax(model(input_data), dim=1)
    return json.dumps({"result": output_tensor.tolist()})


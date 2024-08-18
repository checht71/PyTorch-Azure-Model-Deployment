The codebase for this deployment has 3 main sections. The first is `Deploy ML model on Azure`, which uploads the model to Azure and runs it. The second is `score.py`, which defines how the model will behave when it is deployed. The third is `Predict from Azure Endpoint`, which takes an image from the users computer and sends it to the Azure model. This readme will break down all the relevant info about each section of the code.

There is also a [YouTube video](https://youtu.be/VfTVIXiffBU) explaining the code that this branch was forked from. It is not exactly the same, but it can give you a better idea of what is going on if that is helpful.

# How to Run:
1. Go to Azure portal and sign up if you haven't. Create a `resource group` and a `workspace`.
2. Open `config-template.json`. It should look like this.
```json
{
    "subscription_id": "your-subscription-id",
    "resource_group": "resource-group-name",
    "workspace_name": "workspace-name", <- Create a workspace in Azure portal, copy info here
    "region": "region", <- East US (or another region)
    "model_name": "model_name", <- could be anything
    "service_name": "service_name", <- could be anything
    "weights_path": "path_to_saved_model_dict", <- Where the model is stored locally
    "image_dir": "image_directory", <- Where your images are to test model
    "scoring_uri": "scoring_uri"
}
```
Replace all of the parameters with your info except `scoring_uri`. Rename the file to `config.json`.

3. Open `Deploy model on Azure.ipynb` and run all the cells until you get to **Cleaning up all the created resources**. Then stop.
4. Run `predict.py`. You should get numbers back as your predictions.
5. Go back to `Deploy model on Azure.ipynb`. run the rest of the cells below **Cleaning up all the created resources**.


# In-Depth Explanations:
Below are more in depth explanations of each file and what they do:
## Deploy model on Azure.ipynb
This is a notebook that you can use to deploy your Azure model. This script loads in parameters from a json file where you specify where and how to load and deploy the model. This script will also load the environment that the model was trained in so that it can run properly. After you are done with the model, you can run `delete_service()` in order to end the deployment.

In order to run this notebook properly, edit `config-template.json` with all of the info that you need to deploy the model such as the path to the model and the name of your service.
After that, run every cell in the notebook until you get to __Cleaning up all of the created resources__. The `scoring_uri` is the link to the service that you will send info to. It should be printed out for you and saved to `config.json` if you run the main cells.

## predict.py
This script is to test that the deployment in Azure is working properly and can receive image input. This will be replaced when the data is loaded in from the app or a blob database. For now you can simply run it and it will load your `scoring_uri` from `config.json`

## Score.py
This script defines how to load the model and make predictions using it.
### Functions:
#### init()
This function loads in the model when the AI service is started in Azure. First it defines the model architecture, then it loads in the weights to the model from `config.json`. Then it will check if the device you're using has a CPU or a GPU and set up the model to run on one of these devices (preferably GPU).

```python
def init():
    global model
    """define model architecture"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

	"""load in training info to the model,
		set up device to run it"""
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
```

### pil_images_to_tensor:
this function takes images and transforms them into tensors which the AI model can understand and make predictions on. It also resizes and normalizes the images so that the model runs faster.

```python
def pil_images_to_tensor(pil_images):
    transforms = v2.Compose(
    [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(128, 128)), v2.Normalize((0.5,), (0.5,))])
    tensors = [transforms(img) for img in pil_images] # transform every image
    image_tensor = torch.stack(tensors) # place images in batch
    return image_tensor
```

### run:
this function is called whenever the model receives data. It takes the images which are encoded in json format, converts them to images, sends the images to the `pil_images_to_tensor` function, makes predictions on the images, and returns the predictions in json format.

```python
def run(raw_data):
    data = json.loads(raw_data)['data'] # load data from request
    np_array = np.array(data).squeeze(0)
    pil_image = [Image.fromarray(np_image.astype(np.uint8)) for np_image in np_array] # transform json string to image
    input_data = pil_images_to_tensor(pil_image) # apply transforms to image
    output_tensor = torch.argmax(model(input_data), dim=1) # predict
    return json.dumps({"result": output_tensor.tolist()}) # send back to user
```

# Mission 01

Created: October 22, 2021 4:06 PM

## Mission 1: Build and deploy a model with Vertex AI

**Step 1: Start Cloud Shell**

![Untitled](Mission%2001%20e88401fac28c49cdb38db0ea44ab0165/Untitled.png)

**Step 2: Run the following commands to confirm your authentication and your current working project**

![Untitled](Mission%2001%20e88401fac28c49cdb38db0ea44ab0165/Untitled%201.png)

**Step 3: Create a model and deploy an endpoint. We will use the pre-trained model from `io-vertex-codelab` and get predictions on the deployed endpoint.**

`**deploy.py**`

```python
from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mpg-model-from-vertex",
    artifact_uri="gs://io-vertex-codelab/mpg-model/",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

# Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)
```

`**predict.py**`

```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/208345581856/locations/us-central1/endpoints/5268094460209659904"
)

# A test example we'll send to our model for prediction
test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])
```

Output:

Now it's time to run the [deploy.py](http://deploy.py) file to create an endpoint:

![Untitled](Mission%2001%20e88401fac28c49cdb38db0ea44ab0165/Untitled%202.png)

And run the [predict.py](http://predict.py) to get a prediction from our deployed model endpoint

![Untitled](Mission%2001%20e88401fac28c49cdb38db0ea44ab0165/Untitled%203.png)
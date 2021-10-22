# Mission 01 Custom

Created: October 22, 2021 4:06 PM

## Mission 1 - Custom: Build and deploy a custom mpg-model with a different dataset with Vertex AI

**Step 1: Create a Dockerfile to include all the commands needed to run an image**

`**Dockerfile**`

```bash
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
```

**Step 2: Add a training model into the `mission_01_custom/trainer/train.py` file. The train.py file is already included in the repo.**

**Step 3: Build the container named `mpg-bonus` locally and then push it to Google Container Registry.** 

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled.png)

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%201.png)

**Output:**

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%202.png)

**Step 4: Run a training job on Vertex AI. I am creating custom training via my own custom container on Google Container Registry.**

**Output:**

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%203.png)

**Step 5: Create a model and deploy an endpoint.**

`**deploy.py**`

```python
from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mpg-custom-model",
    artifact_uri="gs://dps-challenge-329218-bucket/mpg/model",
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
    endpoint_name="projects/208345581856/locations/us-central1/endpoints/7456843879111720960"
)

# A test example we'll send to our model for prediction

test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0, 1]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG Custom: ', response[0][0][0])
```

Output:

Now it's time to run the [deploy.py](http://deploy.py) file to create an endpoint 

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%204.png)

And run the [predict.py](http://predict.py) to get a prediction from our deployed model endpoint:

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%205.png)

![Untitled](Mission%2001%20Custom%2077dd57e55dc54dd5bd91a47a192c8ef0/Untitled%206.png)
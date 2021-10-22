# Mission 01 Bonus

Created: October 22, 2021 4:06 PM

## Mission 1 - Bonus: Build and deploy a saleprice-model with a different dataset with Vertex AI

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

**Step 2: Add a training model into the `mission_01_bonus/trainer/train.py` file. The train.py file is already included in the repo.**

**Step 3: Build the container named `mpg-bonus` locally and then push it to Google Container Registry.** 

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled.png)

**Output:**

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled%201.png)

**Step 4: Run a training job on Vertex AI. I am creating custom training via my own custom container on Google Container Registry.**

**Output:**

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled%202.png)

**Step 5: Create a model and deploy an endpoint.**

`**deploy.py**`

```python
from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="saleprice-model",
    artifact_uri="gs://dps-challenge-329218-bucket/saleprice_custom/model",
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
    endpoint_name="projects/208345581856/locations/us-central1/endpoints/7691031059734986752"
)

# A test example we'll send to our model for prediction

test_house = [0.16543080587132408, 0.14661712122274417, 1.0, 0.5, 0.0, 0.0, 0.5555555555555556, 0.6159420289855078,
0.4166666666666643, 0.0, 0.6666666666666666, 1.0, 1.0, 1.0, 0.75, 0.8333333333333333, 0.21598157335223248, 1.0, 0.0,
0.34931506849315064, 0.3330605564648118, 1.0, 1.0, 0.5004589261128959, 0.0, 0.0, 0.41088922381311227, 0.3333333333333333,
0.0, 1.0, 0.0, 0.5, 0.6666666666666666, 1.0, 0.5833333333333334, 0.0, 0.6666666666666666, 1.0, 0.9825870646766168,
0.6666666666666666, 0.5, 0.3413258110014104, 1.0, 1.0, 1.0, 0.0, 0.0, 0.36231884057971014,
0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

response = endpoint.predict([test_house])

print('API response: ', response)

print('Predicted Saleprice: ', response[0][0][0])
```

Output:

Now it's time to run the [deploy.py](http://deploy.py) file to create an endpoint 

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled%203.png)

And run the [predict.py](http://predict.py) to get a prediction from our deployed model endpoint:

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled%204.png)

![Untitled](Mission%2001%20Bonus%20d721a754efe448b2b80ae97bab7f43d5/Untitled%205.png)
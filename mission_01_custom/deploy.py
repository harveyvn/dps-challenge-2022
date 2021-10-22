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
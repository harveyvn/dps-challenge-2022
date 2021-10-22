from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/208345581856/locations/us-central1/endpoints/7456843879111720960"
)

# A test example we'll send to our model for prediction

test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0, 1]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG Custom: ', response[0][0][0])
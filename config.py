"""Configure program configs"""
KEYFILE = 'chris-loves-ai-key.json'
GCP_PROJECT = 'chris-loves-ai'
GS_BUCKET_NAME = 'chris-loves-ai'
KERAS_MODEL_BUCKET = f'gs://{GS_BUCKET_NAME}/wine/keras'
SKLEARN_MODEL_BUCKET = f'gs://{GS_BUCKET_NAME}/wine/sklearn'
MONGO_URL = 'mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb'

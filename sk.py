from sklearn.linear_model import LinearRegression
import subprocess
import data
import pickle
import os
import config
from datetime import datetime
from pathlib import Path


def train_sklearn():
    train_data, train_labels, test_data, test_labels = data.load_data()

    sklearn_version_name = 'v{}'.format(datetime.now().strftime('%Y%m%d_%H%M'))
    sklearn_model_export_path = os.path.join(config.SKLEARN_MODEL_BUCKET, sklearn_version_name)

    scikit_model = LinearRegression().fit(train_data.values, train_labels.values)

    # Export the model to a local file using pickle
    pickle.dump(scikit_model, open('model.pkl', 'wb'))

    # Copy the saved model to Cloud Storage
    # !gsutil cp ./model.pkl gs://chris-loves-ai/sklearn/model.pkl

    dir = Path().absolute()
    model_file = os.path.join(dir, 'model.pkl')
    # subprocess.call(f"gsutil cp model.pkl {sklearn_model_export_path}/model.pkl", shell=True)
    data.upload_file_gs(config.GS_BUCKET_NAME, model_file, f"{sklearn_model_export_path}/model.pkl")

    # Create a new model in our project, you only need to run this once
    # !gcloud ai-platform models create sklearn_wine
    print("gcloud ai-platform models create sklearn_wine")
    subprocess.call("gcloud ai-platform models create sklearn_wine", shell=True)

    print(f"gcloud beta ai-platform versions create {sklearn_version_name} --model=sklearn_wine \
    --origin={sklearn_model_export_path} \
    --runtime-version=1.14 \
    --python-version=3.5 \
    --framework='SCIKIT_LEARN'")

    subprocess.call(f"gcloud beta ai-platform versions create {sklearn_version_name} --model=sklearn_wine \
    --origin={sklearn_model_export_path} \
    --runtime-version=1.14 \
    --python-version=3.5 \
    --framework='SCIKIT_LEARN'", shell=True)

    # Test the model usnig the same example instance from above
    # !gcloud ai-platform predict --model=sklearn_wine --json-instances=predictions.json --version=$SKLEARN_VERSION_NAME
    subprocess.call(f"gcloud ai-platform predict --model=sklearn_wine --json-instances=predictions.json \
    --version={sklearn_version_name}", shell=True)

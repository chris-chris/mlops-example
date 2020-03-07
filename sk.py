import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression, SGDRegressor

import config
import data


def train_sklearn(args):
  """
  Train sklearn models
  :param args:
  :return:
  """
  train_data, train_labels, test_data, test_labels = data.load_data()

  sklearn_version_name = '{}v{}'.format(args.sklearn_model,
                                        datetime.now().strftime('%Y%m%d_%H%M'))
  sklearn_model_export_path = os.path.join(config.SKLEARN_MODEL_BUCKET,
                                           sklearn_version_name)

  if args.sklearn_model == 'linear':
    scikit_model = LinearRegression().fit(train_data.values,
                                          train_labels.values)
  elif args.sklearn_model == 'sgd':
    scikit_model = SGDRegressor(loss=args.loss, alpha=args.lr) \
      .fit(train_data.values, train_labels.values)
  else:
    raise Exception(f'sklearn_model {args.sklearn_model} is not supported')

  test_loss = scikit_model.score(test_data, test_labels)
  test_loss = test_loss
  print('test_loss:', test_loss)
  '''@nni.report_final_result(test_loss)'''

  # Export the model to a local file using pickle
  pickle.dump(scikit_model, open('model.pkl', 'wb'))

  # Copy the saved model to Cloud Storage
  # !gsutil cp ./model.pkl gs://chris-loves-ai/sklearn/model.pkl

  model_path = Path().absolute()
  model_file = os.path.join(model_path, 'model.pkl')
  data.upload_file_gs(config.GS_BUCKET_NAME, model_file,
                      f"{sklearn_model_export_path}/model.pkl")

  # Create a new model in our project, you only need to run this once
  # !gcloud ai-platform models create sklearn_wine
  cmd = "gcloud ai-platform models create sklearn_wine"
  print(cmd)
  subprocess.call(cmd, shell=True)

  cmd = f"gcloud beta ai-platform versions create " \
        f"{sklearn_version_name} --model=sklearn_wine \
    --origin={sklearn_model_export_path} \
    --runtime-version=1.14 \
    --python-version=3.5 \
    --framework='SCIKIT_LEARN'"

  print(cmd)
  subprocess.call(cmd, shell=True)

  cmd = f"gcloud ai-platform predict " \
        f"--model=sklearn_wine --json-instances=predictions.json " \
        f"--version={sklearn_version_name}"
  subprocess.call(cmd, shell=True)

  return test_loss

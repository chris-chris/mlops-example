"""
Load data & Upload models
"""
import subprocess
import pandas as pd
from sklearn.utils import shuffle
from google.cloud import storage
import os


def load_data():
  """
  Load wine quality data for MLOps example.
  :return:
  """
  if not os.path.exists('winequality-white.csv'):
    subprocess.call("wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv' -O winequality-white.csv",
                    shell=True)

    # !wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

  data = pd.read_csv('winequality-white.csv', index_col=False, delimiter=';')
  data = shuffle(data, random_state=4)

  print(data.head())

  labels = data['quality']

  print(labels.value_counts())
  data = data.drop(columns=['quality'])

  train_size = int(len(data) * 0.8)
  train_data = data[:train_size]
  train_labels = labels[:train_size]

  test_data = data[train_size:]
  test_labels = labels[train_size:]

  train_data.head()

  return train_data, train_labels, test_data, test_labels


def upload_file_gs(bucket_name, filepath, gspath):
  """
  Upload model file to Google Storage
  :param bucket_name:
  :param filepath:
  :param gspath:
  :return:
  """
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  gspath = gspath.replace(f'gs://{bucket_name}/', '')

  blob = bucket.blob(gspath)
  blob.upload_from_filename(filepath)

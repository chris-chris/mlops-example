import pandas as pd
import numpy as np
import tensorflow as tf
import witwidget
import os
import pickle

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

import subprocess
from pathlib import Path
from datetime import datetime
import config

dir = Path().absolute()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(dir, config.KEYFILE)

# This has been tested on TF 1.14
print(tf.__version__)


def load_data():

    subprocess.call("wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -O winequality-white.csv'",
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


def train_keras():

    keras_version_name = 'v{}'.format(datetime.now().strftime('%Y%m%d_%H%M'))
    keras_model_export_path = os.path.join(config.KERAS_MODEL_BUCKET, 'keras', keras_version_name)

    train_data, train_labels, test_data, test_labels = load_data()
    # This is the size of the array we'll be feeding into our model for each wine example
    input_size = len(train_data.iloc[0])
    print(input_size)

    model = Sequential()
    model.add(Dense(200, input_shape=(input_size,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()

    model.fit(train_data.values, train_labels.values, epochs=4, batch_size=32, validation_split=0.1)

    # Update these to your own GCP project + model names

    # Add the serving input layer below in order to serve our model on AI Platform
    class ServingInput(tf.keras.layers.Layer):
        # the important detail in this boilerplate code is "trainable=False"
        def __init__(self, name, dtype, batch_input_shape=None):
            super(ServingInput, self).__init__(trainable=False, name=name, dtype=dtype,
                                               batch_input_shape=batch_input_shape)

        def get_config(self):
            return {'batch_input_shape': self._batch_input_shape, 'dtype': self.dtype, 'name': self.name }

    restored_model = model

    serving_model = tf.keras.Sequential()
    serving_model.add(ServingInput('serving', tf.float32, (None, input_size)))
    serving_model.add(restored_model)
    tf.contrib.saved_model.save_keras_model(serving_model, keras_model_export_path)  # export the model to your GCS bucket

    # Configure gcloud to use your project
    # !gcloud config set project $GCP_PROJECT

    subprocess.call(f"gcloud config set project {config.GCP_PROJECT}", shell=True)

    subprocess.call(f"gcloud ai-platform models create keras_wine", shell=True)

    """
    # Deploy the model to Cloud AI Platform
    !gcloud beta ai-platform versions create $KERAS_VERSION_NAME --model keras_wine \
    --origin=$export_path \
    --python-version=3.5 \
    --runtime-version=1.14 \
    --framework='TENSORFLOW'
    """
    cmd = f"""gcloud beta ai-platform versions create {keras_version_name} --model keras_wine \
    --origin={keras_model_export_path} \
    --python-version=3.5 \
    --runtime-version=1.14 \
    --framework='TENSORFLOW'"""
    subprocess.call(cmd, shell=True)

    # prediction = !gcloud ai-platform predict --model=keras_wine --json-instances=predictions.json --version=$KERAS_VERSION_NAME
    # print(prediction)
    subprocess.call(f"gcloud ai-platform predict --model=keras_wine --json-instances=predictions.json --version={keras_version_name}", shell=True)

    print(f"model: keras_wine version: {keras_version_name} path: {keras_model_export_path}")


if __name__ == '__main__':
    train_keras()

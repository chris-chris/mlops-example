import pickle
import config
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from datetime import datetime
import data
import os
import subprocess
from tensorflow.keras import callbacks

import train
import math


def write_log(logs, ex):
    print(logs)
    ex.log_scalar('loss', logs.get('loss'))
    ex.log_scalar('val_loss', logs.get('val_loss'))

def train_keras(args, ex):

    # This has been tested on TF 1.14
    print(tf.__version__)

    keras_version_name = 'v{}'.format(datetime.now().strftime('%Y%m%d_%H%M'))
    keras_model_export_path = os.path.join(config.KERAS_MODEL_BUCKET, keras_version_name)

    train_data, train_labels, test_data, test_labels = data.load_data()
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

    cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: write_log(logs, ex)
    )

    model.fit(train_data.values, train_labels.values, epochs=args.epoch, batch_size=args.batch_size,
              validation_split=0.1,
              callbacks=[cb])
    test_loss = model.evaluate(test_data, test_labels)
    print("final %s" % test_loss)
    '''@nni.report_final_result(test_loss)'''
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
    tf.keras.models.save_model(serving_model, keras_model_export_path)
    # tf.keras.models.save_model(serving_model, keras_model_export_path)  # export the model to your GCS bucket

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

    return test_loss

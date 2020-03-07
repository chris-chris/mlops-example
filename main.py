import logging
import traceback

from flask import jsonify
from googleapiclient import discovery
from werkzeug.exceptions import BadRequest

import config
from codes import *


def predict_via_ai_platform(inputs, model, version=None):
  """prediction rate prediction model via ai platform
  :param inputs: predictable dict. made by data_to_predictable_dict().
  :param model: deployed model at ai-platform..
  :param version: deployed version of model. default=None.
  :return: predicted value. list.
  """

  service = discovery.build('ml', 'v1')
  name = f'projects/{config.GCP_PROJECT}/models/{model}'

  if version is not None:
    name += f'/versions/{version}'

  response = service.projects().predict(
      name=name,
      body=inputs
  ).execute()

  if 'error' in response:
    raise RuntimeError(response['error'])

  return response['predictions']


def predict_wine(request):
  """
  Predict wine quality Functions API
  :param request:
  :return:
  """
  try:
    """
    :param request: request json object has three values.
                    request {
                        model: "model name",
                        version: "version name of model",
                    }
    functions test sample:
    {
        "inputs": [[7.8, 0.21, 0.49, 1.2, 0.036,
         20.0, 99.0, 0.99, 3.05, 0.28, 12.1]], 
        "model": "keras_wine", 
        "version": "v20191115_1117", 
    }
    """

    parameters = [
        MODEL,
        VERSION,
        INPUTS,
    ]

    if not request.get_json():
      raise BadRequest('invalid request body : body should be json')

    json_request = request.get_json()
    for parameter in parameters:
      if not json_request.get(parameter) is None:
        continue
      raise BadRequest('invalid request parameter: %s' % parameter)

    model = json_request.get(MODEL)
    version = json_request.get(VERSION)
    inputs = json_request.get(INPUTS)

    data = predict_via_ai_platform(inputs, model, version)

    res = {'code': 200, 'message': 'Success', 'data': data}
    return jsonify(res)
  except Exception as e:
    traceback.print_exc()
    var = traceback.format_exc()
    message = f'Exception {e}\n{var}'
    logging.error(message)
    res = {'code': 500, 'message': message, 'data': None}
    return jsonify(res)


if __name__ == '__main__':
  test_model = 'keras_wine'
  test_version = 'v20191115_1722'
  test_inputs = [[7.8, 0.21, 0.49, 1.2, 0.036,
                  20.0, 99.0, 0.99, 3.05, 0.28, 12.1]]
  instances = {
      'instances': test_inputs
  }
  prediction = predict_via_ai_platform(instances, test_model, test_version)
  print(f"prediction: {prediction}")

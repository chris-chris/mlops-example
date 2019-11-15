curl -X POST \
  https://asia-northeast1-chris-loves-ai.cloudfunctions.net/predict_wine \
  -H 'Content-Type: application/json' \
  -d '{
  "model":"sklearn_wine",
  "version":"linearv20191115_1703",
  "inputs":{
        "instances": [[7.8, 0.21, 0.49, 1.2, 0.036, 20.0, 99.0, 0.99, 3.05, 0.28, 12.1]]
    }
}'
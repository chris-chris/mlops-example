gcloud \
  beta functions \
  deploy predict_wine \
  --runtime python37 \
  --trigger-http \
  --project chris-loves-ai \
  --region asia-northeast1 \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=chris-loves-ai-key.json

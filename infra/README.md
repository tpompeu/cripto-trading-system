### 1.
aws cloudformation deploy \
  --template-file infra/pipeline.yaml \
  --stack-name crypto-trading-pipeline \
  --parameter-overrides \
      GitHubOwner=tpompeu \
      GitHubRepo=cripto-trading-system \
      GitHubBranch=main \
      ArtifactsBucket=crypto-trading-pipeline-artifacts-957280467604 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile bts
### 1.
aws s3 mb s3://crypto-trading-system-pipeline-artifacts \
  --region ap-southeast-1 \
  --profile cts

### 2.
aws cloudformation deploy \
  --template-file infra/pipeline.yaml \
  --stack-name crypto-trading-pipeline \
  --parameter-overrides \
      GitHubOwner=tpompeu \
      GitHubRepo=cripto-trading-system \
      GitHubBranch=main \
      ArtifactsBucket=crypto-trading-system-pipeline-artifacts \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile cts
### 1. 
aws cloudformation deploy \
  --template-file iam.yaml \
  --stack-name crypto-trading-iam \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

### 2.
aws cloudformation deploy \
  --template-file infra/pipeline.yaml \
  --stack-name crypto-trading-pipeline \
  --parameter-overrides \
      CodeBuildRoleArn=$(aws cloudformation describe-stacks --stack-name crypto-trading-iam --query "Stacks[0].Outputs[?OutputKey=='CodeBuildRoleArn'].OutputValue" --output text --profile bts) \
      CloudFormationRoleArn=$(aws cloudformation describe-stacks --stack-name crypto-trading-iam --query "Stacks[0].Outputs[?OutputKey=='CloudFormationRoleArn'].OutputValue" --output text --profile bts) \
      GitHubOwner=tpompeu \
      GitHubRepo=cripto-trading-system \
      GitHubBranch=main \
      ArtifactsBucket=crypto-trading-pipeline-artifacts-957280467604 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile bts
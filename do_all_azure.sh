#! /bin/bash

# Build the container
docker build -t rpaff/cs553_example_azure .

# Push the container to docker hub
docker push rpaff/cs553_example_azure

# Run the container on Azure
az containerapp create \
  --name cs553-cli-example-1 \
  --resource-group rcpaffenroth-rg \
  --environment managedenvironment-rcpaffenroth-rg \
  --image rpaff/cs553_example_azure \
  --ingress external \
  --target-port 7860

# Get the URL
az containerapp show \
  --name cs553-cli-example-1 \
  --resource-group rcpaffenroth-rg \
  --query properties.configuration.ingress.fqdn

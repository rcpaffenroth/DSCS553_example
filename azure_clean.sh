#! /bin/bash

# Build the container
az containerapp delete \
  --name cs553-cli-example-1 \
  --resource-group rcpaffenroth-rg

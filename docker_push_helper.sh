#!/bin/bash

az acr login --name mlopsfinal

docker build -t mlopsfinal.azurecr.io/base:latest -f DependencyDockerfile .

docker push mlopsfinal.azurecr.io/base:latest

docker build -t mlopsfinal.azurecr.io/digits:latest -f FinalDockerfile .

docker push mlopsfinal.azurecr.io/digits:latest

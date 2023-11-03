#!/bin/bash

podman build -t digit_classification .

mkdir model_view

ls -lh model_view

podman run -v ./model_view:/digits/models digit_classification

ls -lh model_view

rm -rf model_view

#!/bin/bash

docker build -t digit_classification .

mkdir model_view

ls -lh model_view

docker run -v ./model_view:/digits/models digit_classification

ls -lh model_view

#rm -rf model_view

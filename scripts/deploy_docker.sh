#!/bin/bash

cd $PROJECT_DIR
cd api
docker build -t my_ml_service .
cd $PROJECT_DIR
docker run --rm -p 5152:8080 my_ml_service

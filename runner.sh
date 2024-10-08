#!/bin/bash
python -m preprocess.vectoriser --model_name $1 --dataset ilpcr
python -m preprocess.vectoriser --model_name $1 --dataset coliee
python -m preprocess.vectoriser --model_name $1 --dataset irled
python -m preprocess.vectoriser --model_name $1 --dataset muser
python -m preprocess.vectoriser --model_name $1 --dataset ecthr
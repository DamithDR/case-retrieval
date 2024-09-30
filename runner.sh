#!/bin/bash
python -m preprocess.vectoriser --model_name dunzhang/stella_en_1.5B_v5 --dataset ilpcr
python -m preprocess.vectoriser --model_name dunzhang/stella_en_1.5B_v5 --dataset coliee
python -m preprocess.vectoriser --model_name dunzhang/stella_en_1.5B_v5 --dataset irled
python -m preprocess.vectoriser --model_name dunzhang/stella_en_1.5B_v5 --dataset muser
python -m preprocess.vectoriser --model_name dunzhang/stella_en_1.5B_v5 --dataset ecthr
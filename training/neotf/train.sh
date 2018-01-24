#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
/usr/bin/python3 net_to_model.py save/best.txt
/usr/bin/python3 parse.py save/leelaz-model-0

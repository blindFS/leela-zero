# Additional Dependencies

* python 3.5+ env with functioning tensorflow-gpu
* [flask](http://flask.pocoo.org/)

# Quickstart Guide

1. Read `config.py` first.
2. copy previous network file, e.g. `../../weights.txt` to `SAVE_DIR/best.txt`
3. Start `server.py`, set 'host' to '0.0.0.0' for remote access.
4. At client side, use `../../neogtp/datagen.sh` to generate and upload traning data, customize the command before start.
5. Compile and copy the executable `leelaz` and `validation` from `../../src`, `../../validation` to current directory.
6. Run `train.sh` 
7. `tensorboard --logdir=./leelalogs` provides visualization for fine tuning.

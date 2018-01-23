#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
from tfprocess import TFProcess
from config import leela_conf

def get_weights(fname):
    weights = []
    with open(fname, 'r') as f:
        for e, line in enumerate(f):
            if e == 0:
                #Version
                print("Version", line.strip())
                if line != '1\n':
                    raise ValueError("Unknown version {}".format(line.strip()))
            else:
                weights.append(list(map(float, line.split(' '))))
            if e == 2:
                channels = len(line.split(' '))
                print("Channels", channels)
        blocks = e - (4 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks /= 8
        print("Blocks", blocks)
    return weights

if __name__ == '__main__':
    gpu_num = 2
    x = [[
        tf.placeholder(tf.float32, [None, 18, 19 * 19]),
        tf.placeholder(tf.float32, [None, 362]),
        tf.placeholder(tf.float32, [None, 1])
    ] for _ in range(gpu_num)]

    tfprocess = TFProcess(x)
    tfprocess.replace_weights(get_weights(sys.argv[1]))
    path = os.path.join(leela_conf.SAVE_DIR, "leelaz-model")
    print("saved to: ", path)
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)

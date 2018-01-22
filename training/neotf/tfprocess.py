#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import time
import tensorflow as tf
from config import leela_conf

def weight_variable(shape, key):
    return tf.get_variable("%s-v" % key, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())

# Bias weights for layers not followed by BatchNorm
def bias_variable(shape, key):
    return tf.get_variable("%s-b" % key, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())

# No point in learning bias weights as they are cancelled
# out by the BatchNorm layers's mean adjustment.
def bn_bias_variable(shape, key):
    return tf.get_variable("%s-bn" % key, shape=shape,
                           initializer=tf.constant_initializer(0), trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class TFProcess:
    def __init__(self, next_batch):
        with tf.device('/cpu:0'):
            self.session = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True)))

            # For exporting
            self.weights = []

            #  opt_op = tf.train.MomentumOptimizer(
                #  learning_rate=leela_conf.LR, momentum=0.9, use_nesterov=True)
            opt_op = tf.train.GradientDescentOptimizer(leela_conf.LR)

            tower_grads = []
            accuracies = []
            policy_losses = []
            mse_losses = []
            reg_terms = []
            # TF variables
            self.next_batch = next_batch
            self.global_step = tf.get_variable('global_step',
                                          [], initializer=tf.constant_initializer(0),
                                          trainable=False)
            self.training = tf.placeholder(tf.bool)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(len(next_batch)):
                    self.i = i
                    self.batch_norm_count = 0
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope("res-go") as scope:
                            x = next_batch[i][0] # tf.placeholder(tf.float32, [None, 18, 19 * 19])
                            y_ = next_batch[i][1] # tf.placeholder(tf.float32, [None, 362])
                            z_ = next_batch[i][2] # tf.placeholder(tf.float32, [None, 1])
                            y_conv, z_conv = self.construct_net(x)

                            # Calculate loss on policy head
                            cross_entropy = \
                                tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                        logits=y_conv)
                            policy_loss = tf.reduce_mean(cross_entropy)

                            # Loss on value head
                            mse_loss = \
                                tf.reduce_mean(tf.squared_difference(z_, z_conv))

                            # Regularizer
                            regularizer = tf.contrib.layers.l2_regularizer(scale=leela_conf.REGULARIZER_SCALE)
                            reg_variables = tf.trainable_variables()
                            reg_term = \
                                tf.contrib.layers.apply_regularization(regularizer, reg_variables)

                            loss = leela_conf.POLICY_LOSS_WEIGHT * policy_loss \
                                + leela_conf.MSE_LOSS_WEIGHT * mse_loss + reg_term

                            correct_prediction = \
                                tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                            correct_prediction = tf.cast(correct_prediction, tf.float32)
                            local_acc = tf.reduce_mean(correct_prediction)

                            policy_losses.append(policy_loss)
                            mse_losses.append(mse_loss)
                            reg_terms.append(reg_term)
                            accuracies.append(local_acc)

                            tf.get_variable_scope().reuse_variables()

                            self.update_ops = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            grad = opt_op.compute_gradients(loss)
                            tower_grads.append(grad)

            #  with tf.control_dependencies(self.update_ops):
            #  self.train_op = \
            #  opt_op.minimize(loss, global_step=self.global_step)
            grads = average_gradients(tower_grads)
            self.train_op = opt_op.apply_gradients(grads, global_step=self.global_step)

            self.accuracy = tf.reduce_mean(accuracies)
            self.policy_loss = tf.reduce_mean(policy_losses)
            self.mse_loss = tf.reduce_mean(mse_losses)
            self.reg_term = tf.reduce_mean(reg_terms)

            self.avg_policy_loss = []
            self.avg_mse_loss = []
            self.avg_reg_term = []
            self.time_start = None

            # Summary part
            self.test_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), "leelalogs/test"), self.session.graph)
            self.train_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), "leelalogs/train"), self.session.graph)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            self.session.run(self.init)

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            # Keyed batchnorm weights
            if isinstance(weights, str):
                print(weights)
                work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                new_weight = tf.constant(new_weights[e])
                self.session.run(tf.assign(work_weights, new_weight))
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                print(shape)
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                print(shape)
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                print(weights.shape)
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(weights.assign(new_weight))
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.session, file)

    def save(self, steps, path):
        save_path = self.saver.save(self.session, path, global_step=steps)
        print("Model saved in file: {}".format(save_path))
        leela_path = path + "-" + str(steps) + ".txt"
        self.save_leelaz_weights(leela_path)
        print("Leela weights saved to {}".format(leela_path))
        return save_path
    
    def info(self, steps):
        time_end = time.time()
        speed = 0
        if self.time_start:
            elapsed = time_end - self.time_start
            speed = leela_conf.BATCH_SIZE * (100.0 / elapsed)
        avg_policy_loss = np.mean(self.avg_policy_loss or [0])
        avg_mse_loss = np.mean(self.avg_mse_loss or [0])
        avg_reg_term = np.mean(self.avg_reg_term or [0])
        print("step {}, policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
            steps, avg_policy_loss, avg_mse_loss, avg_reg_term,
            avg_policy_loss + avg_mse_loss + avg_reg_term,
            speed))
        train_summaries = tf.Summary(value=[
            tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
            tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
        self.train_writer.add_summary(train_summaries, steps)
        self.time_start = time_end
        self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []

    def eval(self, steps):
        sum_accuracy = 0
        sum_mse = 0
        for _ in range(0, 10):
            train_accuracy, train_mse, _ = self.session.run(
                [self.accuracy, self.mse_loss, self.next_batch],
                feed_dict={self.training: False})
            sum_accuracy += train_accuracy
            sum_mse += train_mse
        sum_accuracy /= 10.0
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * 10.0)
        test_summaries = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
            tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
        self.test_writer.add_summary(test_summaries, steps)
        print("step {}, training accuracy={:g}%, mse={:g}".format(
            steps, sum_accuracy*100.0, sum_mse))
        self.save(steps, os.path.join(leela_conf.SAVE_DIR, "leelaz-model"))

    def process(self):
        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op,
                self.next_batch],
            feed_dict={self.training: True})
        steps = tf.train.global_step(self.session, self.global_step)
        # Keep running averages
        # XXX: use built-in support like tf.moving_average_variables?
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss = mse_loss / 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % leela_conf.INFO_STEP_INTERVAL == 0:
            self.info(steps)
        # Ideally this would use a seperate dataset and so on...
        if steps % leela_conf.EVAL_STEP_INTERVAL == 0:
            self.eval(steps)
            return True
        return False

    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        weight_key = self.get_batchnorm_key()
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        with tf.variable_scope(weight_key, reuse=tf.AUTO_REUSE):
            W_conv = weight_variable([filter_size, filter_size,
                                      input_channels, output_channels], weight_key)
            b_conv = bn_bias_variable([output_channels], weight_key)
            if self.i == 0:
                self.weights.append(W_conv)
                self.weights.append(b_conv)
                self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
                self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

            h_bn = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        weight_key_1 = self.get_batchnorm_key()
        with tf.variable_scope(weight_key_1, reuse=tf.AUTO_REUSE):
            W_conv_1 = weight_variable([3, 3, channels, channels], weight_key_1)
            b_conv_1 = bn_bias_variable([channels], weight_key_1)

            if self.i == 0:
                self.weights.append(W_conv_1)
                self.weights.append(b_conv_1)
                self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
                self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)

        # Second convnet
        weight_key_2 = self.get_batchnorm_key()
        with tf.variable_scope(weight_key_2, reuse=tf.AUTO_REUSE):
            W_conv_2 = weight_variable([3, 3, channels, channels], weight_key_2)
            b_conv_2 = bn_bias_variable([channels], weight_key_2)
            if self.i == 0:
                self.weights.append(W_conv_2)
                self.weights.append(b_conv_2)
                self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
                self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))

        return h_out_2

    def construct_net(self, planes):
        # Network structure
        RESIDUAL_FILTERS = leela_conf.RESIDUAL_FILTERS
        RESIDUAL_BLOCKS = leela_conf.RESIDUAL_BLOCKS

        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=RESIDUAL_FILTERS)
        # Residual tower
        for _ in range(0, RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, RESIDUAL_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=RESIDUAL_FILTERS,
                                   output_channels=2)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2*19*19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1], "policy-head")
        b_fc1 = bias_variable([(19 * 19) + 1], "policy-head")
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        if self.i == 0:
            self.weights.append(W_fc1)
            self.weights.append(b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=RESIDUAL_FILTERS,
                                   output_channels=1)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19*19])
        W_fc2 = weight_variable([19 * 19, 256], "value-head0")
        b_fc2 = bias_variable([256], "value-head0")
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([256, 1], "value-head1")
        b_fc3 = bias_variable([1], "value-head1")
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        if self.i == 0:
            self.weights.append(W_fc2)
            self.weights.append(b_fc2)
            self.weights.append(W_fc3)
            self.weights.append(b_fc3)

        return h_fc1, h_fc3

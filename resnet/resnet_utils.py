# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

DATA_FORMAT = "NCHW"  # to match data format for mxnet

def resnet_arg_scope(is_training=True, weight_decay=0.0001,
                     batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def stem_7x7(net, scope="C1"):
    with tf.variable_scope(scope):
        net = slim.conv2d(net, num_outputs=64, kernel_size=[7, 7], stride=2,
                          padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                          scope="conv0")

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="SAME", data_format=DATA_FORMAT)
        return net

def stem_stack_3x3(net, input_channel=32, scope="C1"):
    with tf.variable_scope(scope):
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                          padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                          padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                          padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv2')
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="SAME", data_format=DATA_FORMAT)
        return net


def bottleneck_v1b(input_x, base_channel, stride=1, projection=False, avg_down=True):
    '''
    for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
    '''
    net = slim.conv2d(input_x, num_outpus=base_channel, kernel_size=[1, 1], stride=1,
                      padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                      scope='conv0')

    net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                      padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                      scope='conv1')

    net = slim.conv2d(net, num_outpus=base_channel * 4, kernel_size=[1, 1], stride=1,
                      padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                      activation_fn=None, scope='conv2')
    # Note that : gamma in the last conv should be init with 0.
    # But we just reload params from mxnet, so don't specific batch norm initializer
    if projection:
        if stride == 2:
            if avg_down:  # design for resnet_v1d
                shortcut = slim.avg_pool2d(input_x, kernel_size=[2, 2], stride=2, padding="SAME",
                                           data_format=DATA_FORMAT)
                shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                       stride=1, padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                                       scope='shortcut')
                # shortcut should have batch norm.
            else:
                shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                       stride=2, padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                                       scope='shortcut')
        elif stride == 1:
            shortcut = slim.conv2d(net, num_outputs=base_channel*4, kernel_size=[3, 3], stride=1,
                                   padding="SAME", biases_initializer=None, data_format=DATA_FORMAT,
                                   scope='shortcut')
        else:
            raise ValueError("stride must be 1 or 2 in bottleneck")
    else:
        shortcut = input_x

    net = net + shortcut
    net = slim.relu(net, data_format=DATA_FORMAT)
    return net


def make_block(net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
    with tf.variable_scope(scope):
        first_stride = 2 if spatial_downsample else 1
        with tf.variable_scope("bottleneck_0"):
            net = bottleneck_v1b(input_x=net, base_channel=base_channel,
                                 stride=first_stride, avg_down=avg_down, projection=True)
        for i in range(1, bottleneck_nums):
            with tf.variable_scope("bottleneck_%d" % i):
                net = bottleneck_v1b(input_x=net, base_channel=base_channel,
                                     stride=1, avg_down=avg_down, projection=False)
        return net


def get_resnet_v1_d_base(input_x, scope="resnet_v1_50", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    feature_dict = {}
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=not freeze[0])):
            net = stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
            feature_dict["C1"] = net
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True
            with slim.arg_scope(resnet_arg_scope(is_training=not freeze[i-2])):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=True, spatial_downsample=spatial_downsample)
                feature_dict["C%d" % i] = net

    return net, feature_dict

def get_resnet_v1_d(input_x, scope="resnet_v1_50", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True, num_cls=1000, dropout=False):

    net = get_resnet_v1_d_base(input_x=input_x, scope=scope, bottleneck_nums=bottleneck_nums, base_channels=base_channels,
                               freeze=freeze, is_training=is_training)
    with tf.variable_scope(scope):
        # net shape : [B, C, H, W]
        if DATA_FORMAT.strip() == "NCHW":
            net = tf.reduce_mean(net, axis=[2, 3], name="global_avg_pooling",
                                 keep_dims=True)  # [B, C, 1, 1]
        elif DATA_FORMAT.strip() == "NHWC":
            net = tf.reduce_mean(net, axis=[1, 2], name="global_avg_pooling",
                                 keep_dims=True)  # [B, 1, 1, C]
        else:
            raise ValueError("Data Format Erro...")

        net = slim.flatten(net, scope='flatten')
        if dropout:
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training)
        net = slim.fully_connected(net, num_outputs=num_cls, activation_fn=None, scope='logits')
        return net







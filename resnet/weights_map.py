# -*-coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)


tf_mxnet_map = {"C1/conv0/weights": "conv1.0.weight",
                "C1/conv0/BatchNorm/beta": "conv1.1.beta",
                "C1/conv0/BatchNorm/gamma": "conv1.1.gamma",
                "C1/conv0/BatchNorm/moving_mean": "conv1.1.running_mean",
                "C1/conv0/BatchNorm/moving_variance": "conv1.1.running_var",
                "C1/conv1/weights": "conv1.3.weight",
                "C1/conv2/weights": "conv1.6.weight"}

tf_mxnet_prefix_map = {"weights": "weight",
                       "moving_mean": "running_mean",
                       "moving_variance": "running_var",
                       "beta": "beta",
                       "gamma": "gamma"}
def update_C1():

    tf_prefix = "C1/conv1/"
    for key in tf_mxnet_prefix_map.keys():
        if key != 'weights':
            tf_mxnet_map[tf_prefix+key] = "conv1.4." + tf_mxnet_prefix_map[key]

    tf_prefix = "C1/conv2/"
    for key in tf_mxnet_prefix_map.keys():
        if key != 'weights':
            tf_mxnet_map[tf_prefix+key] = "bn1." + tf_mxnet_prefix_map[key]

def update_C2(bottleneck_num):
    for i in range(bottleneck_num):
        for j in range(3):
            tf_prefix = "C2/bottleneck_%d/conv%d/" % (i, j)
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix+key] = "layer1.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer1.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
        if i==0:
            tf_prefix = "C2/bottleneck_%d/shortcut/" % i
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix + key] = "layer1.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer1.%d.downsample.2." % i + tf_mxnet_prefix_map[key]

def update_C3(bottleneck_num):

    for i in range(bottleneck_num):
        for j in range(3):
            tf_prefix = "C3/bottleneck_%d/conv%d/" % (i, j)
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix+key] = "layer2.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer2.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
        if i == 0:
            tf_prefix = "C3/bottleneck_%d/shortcut/" % i
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix + key] = "layer2.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer2.%d.downsample.2." % i + tf_mxnet_prefix_map[key]


def update_C4(bottleneck_num):
    for i in range(bottleneck_num):
        for j in range(3):
            tf_prefix = "C4/bottleneck_%d/conv%d/" % (i, j)
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix+key] = "layer3.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer3.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
        if i == 0:
            tf_prefix = "C4/bottleneck_%d/shortcut/" % i
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix + key] = "layer3.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer3.%d.downsample.2." % i + tf_mxnet_prefix_map[key]

def update_C5(bottleneck_num):
    for i in range(bottleneck_num):
        for j in range(3):
            tf_prefix = "C5/bottleneck_%d/conv%d/" % (i, j)
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix+key] = "layer4.%d.conv%d." % (i, j+1) + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer4.%d.bn%d." % (i, j + 1) + tf_mxnet_prefix_map[key]
        if i == 0:
            tf_prefix = "C5/bottleneck_%d/shortcut/" % i
            for key in tf_mxnet_prefix_map.keys():
                if key == 'weights':
                    tf_mxnet_map[tf_prefix + key] = "layer4.%d.downsample.1." % i + tf_mxnet_prefix_map[key]
                else:
                    tf_mxnet_map[tf_prefix + key] = "layer4.%d.downsample.2." % i + tf_mxnet_prefix_map[key]

def update_logitis():
    tf_mxnet_map["logits/weights"] = 'fc.weight'
    tf_mxnet_map["logits/biases"] = 'fc.bias'
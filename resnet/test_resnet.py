# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from mxnet import nd, image
import numpy as np
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval
from resnet import build_resnet, create_resotre_op
import os


MODEL_NAME = 'resnet50_v1b'
# Mxnet_Weights_PATH = '../mxnet_weights/resnet50_v1d-117a384e.params'
Mxnet_Weights_PATH = '../mxnet_weights/resnet50_v1b-0ecdba34.params'

def mxnet_process_img(path):
    # Load Images
    img = image.imread(path)

    # Transform
    img = transform_eval(img)

    print (img.shape)

    img_arr = img.asnumpy()
    if len(img_arr) == 3:
        img_arr = np.expand_dims(img_arr, axis=0)
    img_tf_tensor = tf.constant(img_arr)
    return img, img_tf_tensor

def mxnet_infer(img):

    model_name = MODEL_NAME
    net = get_model(model_name, pretrained=True)
    net.load_parameters(Mxnet_Weights_PATH)
    pred = net(img)

    print (pred.shape, pred.dtype)

    pred = pred.asnumpy()
    return pred


def tf_infer(img):

    pred_tensor = build_resnet(img_batch=img, scope=MODEL_NAME,
                               is_training=False, freeze_norm=True, num_cls=1000)

    restore_op = create_resotre_op(MODEL_NAME, Mxnet_Weights_PATH)
    with tf.Session() as sess:
        sess.run(restore_op)
        pred = sess.run(pred_tensor)
    sess.close()

    return pred


def cal_erro(img_path):

    mxnet_img, tf_img = mxnet_process_img(img_path)

    mxnet_pred = mxnet_infer(mxnet_img)

    mxnet_pred = np.squeeze(mxnet_pred, axis=0)
    tf_pred = tf_infer(tf_img)
    tf_pred = np.squeeze(tf_pred, axis=0)
    assert mxnet_pred.shape == tf_pred.shape, "mxnet_pred shape Do Not equal with tf_pred shape"

    argmax_mxnet = np.argmax(mxnet_pred)
    argmax_tf = np.argmax(tf_pred)

    erro = np.linalg.norm(tf_pred-mxnet_pred)
    for i, (m, t) in enumerate(zip(mxnet_pred, tf_pred)):
        if i == 5:
            break
        print ("mxnet|tf==>>{} | {} ".format(m, t))

    print (erro)
    print (erro/np.linalg.norm(mxnet_pred))
    print ("argmax_mxnet: {} || tf_argmx: {}".format(argmax_mxnet, argmax_tf))


if __name__ == '__main__':

    cal_erro(img_path='../demo_img/person.jpg')
    print ("20*++")
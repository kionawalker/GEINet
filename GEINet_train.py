# _*_ coding:utf-8 _*_

import numpy as np
import os
import glob
from PIL import Image
import cv2
import chainer
import cupy as cp
from chainer import Serializer, training, reporter, iterators, Function, cuda ,initializers
from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer.training import extensions
from chainer import serializers
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.functions.evaluation import accuracy
import six

import sys
sys.path.append('/home/common-ns/PycharmProjects/Prepare')
from tools import load_GEI
sys.path.append('/home/common-ns/PycharmProjects/models')
from GEINet import GEINet
from sys import argv


# train model
def train(train_dir):

    train1 = load_GEI(path_dir=train_dir, mode=True)

    model = L.Classifier(GEINet())

    model.to_gpu()


    Dt1_train_iter = iterators.SerialIterator(train1, batch_size=239, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.02, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

    # updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'second': 1})
    updater = training.StandardUpdater(model, Dt1_train_iter, optimizer, device=0)
    epoch = 6250

    trainer = training.Trainer(updater, (epoch, 'epoch'),
                               out='/home/wutong/Setoguchi/chainer_files/result')

    # trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.ExponentialShift(attr='lr', rate=0.56234), trigger=(1250, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='SFDEI_log', trigger=(50, "epoch")))
    trainer.extend(extensions.snapshot(), trigger=(1250, 'epoch'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_{.updater.epoch}'), trigger=(1250, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch',
                                           'accuracy',
                                           'loss']))
    # 'validation/main/accuracy']),
    # trigger=(1, "epoch"))
    trainer.extend(extensions.dump_graph(root_name="loss", out_name="multi_modal.dot"))
    trainer.extend(extensions.PlotReport(["loss"]), trigger=(20, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # Run the trainer
    trainer.run()


if __name__ == '__main__':

    train(argv[0])
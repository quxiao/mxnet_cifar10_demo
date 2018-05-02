# -*- coding: utf-8 -*-

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

# 调用SDK API进行训练实例的创建和状态更新
from ava.train import base as train
# 训练指标监控上报
from ava.monitor import mxnet as mxnet_monitor
from ava.utils import utils

if __name__ == '__main__':
    # download data
    #(train_fname, val_fname) = download_cifar10()
    
    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-train', help='training data, recdio file', type=str)
    parser.add_argument(
        '--data-val', help='validation data, recdio file', type=str)

     
    # 在一个训练任务的训练环境中，每一次训练被称为一个“训练实例”
    train_ins = train.TrainInstance()

    # 添加监控
    snapshot_prefix = train_ins.get_snapshot_base_path() + "/snapshot"
    snapshot_interval_epochs = 1
    #snapshot_interval_epochs = params.get_value(
    #        "intervals.snapshotIntervalEpochs", default=1)
    
    # add CALLBACK
    batch_end_cb = train_ins.get_monitor_callback(
        "mxnet",
        batch_size=128,  # args.batch_size
        batch_freq=10)
    args.batch_end_callback = batch_end_cb
    # 测试
    actual_batch_size = 128 * 2
    batch_of_epoch = utils.ceil_by_level(
            float(utils.get_sampleset_num() / actual_batch_size))
    epoch_end_cb = [
        # mxnet default epoch callback
        mx.callback.do_checkpoint(
            snapshot_prefix, snapshot_interval_epochs),
        train_ins.get_epoch_end_callback(
            "mxnet", batch_of_epoch=batch_of_epoch,
            epoch_interval=snapshot_interval_epochs, other_files=[])
    ] 

    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network='resnet',
        num_layers=50,
        # data
        num_classes=10,
        num_examples=50000,
        image_shape='3,28,28',
        pad_size=4,
        # train
        batch_size=128,
        num_epochs=300,
        lr=.05,
        lr_step_epochs='200,250'
        epoch_end_callback=epoch_end_cb,
        batch_end_callback=batch_end_cb
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)

    # 训练结束，更新训练实例的状态，err_msg为空时表示训练正常结束，
    # 不为空表示训练异常结束
    train_ins.done(err_msg=err_msg)

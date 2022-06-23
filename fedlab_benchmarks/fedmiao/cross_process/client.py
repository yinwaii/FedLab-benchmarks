# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 17:38:56
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 17:39:01

import sys
import torch
import argparse
import os

from setting import get_model, get_dataset
from torch import nn, optim
from fedlab.core.network import DistNetwork
from fedlab.core.client.manager import PassiveClientManager
from fedlab.utils.logger import Logger
from fedlab.utils.functional import get_best_gpu

sys.path.append("../")
from fedamp_trainer import FedAmpTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distbelief training example")

    parser.add_argument("--ip", type=str)

    parser.add_argument("--port", type=str)

    parser.add_argument("--world_size", type=int)

    parser.add_argument("--rank", type=int)

    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--epoch", type=int, default=5)

    parser.add_argument("--dataset", type=str)

    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument("--gpu", type=str, default="0,1,2,3")

    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument(
        "--alphaK", type=float, default=0.005
    )  # vaild value should be in range [0, 1] and mod 0.1 == 0

    parser.add_argument(
        "--optimizer", type=str,
        default="sgd")  # valid value: {"sgd", "adam", "rmsprop"}

    parser.add_argument(
        "--lamda", type=float,
        default=5e-7)  # recommended value: {0.001, 0.01, 0.1, 1.0}

    args = parser.parse_args()

    if args.gpu != "-1":
        args.cuda = True
        device = torch.device(get_best_gpu())
    else:
        args.cuda = False
        device = torch.device("cpu")

    model = get_model(args).to(device)
    trainloader, testloader = get_dataset(args)
    optimizer_dict = dict(sgd=optim.SGD,
                          adam=optim.Adam,
                          rmsprop=optim.RMSprop)
    optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=args.rank,
        ethernet=args.ethernet,
    )

    LOGGER = Logger(log_name="client " + str(args.rank))

    trainer = FedAmpTrainer(
        model=model,
        data_loader=trainloader,
        epochs=args.epoch,
        optimizer=optimizer,
        criterion=criterion,
        args=args,
        cuda=args.cuda,
        logger=LOGGER,
    )

    manager_ = PassiveClientManager(trainer=trainer,
                                    network=network,
                                    logger=LOGGER)
    manager_.run()

# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 17:39:06
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 17:39:18

from handler import FedAmpHandler, FedAmpManager
from setting import get_model, get_dataset
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
import sys
import argparse
sys.path.append("..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL server example")

    parser.add_argument("--ip", type=str)

    parser.add_argument("--port", type=str)

    parser.add_argument("--world_size", type=int)

    parser.add_argument("--round", type=int, default=5)

    parser.add_argument("--dataset", type=str)

    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument("--sample", type=float, default=1)

    parser.add_argument(
        "--alphaK", type=float, default=0.005
    )  # vaild value should be in range [0, 1] and mod 0.1 == 0

    parser.add_argument(
        "--sigma", type=float, default=0.1
    )  # recommended value: {0.001, 0.01, 0.1, 1.0}

    args = parser.parse_args()

    model = get_model(args)
    _, test_loader = get_dataset(args)
    LOGGER = Logger(log_name="server")
    handler = FedAmpHandler(model,
                            test_loader=test_loader,
                            global_round=args.round,
                            logger=LOGGER,
                            sample_ratio=args.sample)
    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=0,
        ethernet=args.ethernet,
    )
    manager_ = FedAmpManager(handler=handler,
                             network=network,
                             logger=LOGGER)
    manager_.run()

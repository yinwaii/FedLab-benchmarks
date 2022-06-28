# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 16:44:38
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 16:45:58

# python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid

import sys

import math
import argparse
import os
import torch
import random
from copy import deepcopy
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu
from fedlab.utils.dataset.sampler import SubsetSampler
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

sys.path.append("../")

from models.cnn import CNN_MNIST
from fedamp_trainer import FedAmpTrainer
from persistence import save_model, load_model

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)

parser.add_argument("--sample_ratio", type=float, default=0.1)

parser.add_argument("--batch_size", type=int, default=100)

parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--epochs", type=int, default=5)

parser.add_argument("--partition", type=str, default="iid")

parser.add_argument("--round", type=int, default=10)

parser.add_argument(
    "--alphaK", type=float, default=10000
)  # vaild value should be in range [0, 1] and mod 0.1 == 0

parser.add_argument(
    "--optimizer", type=str, default="adam"
)  # valid value: {"sgd", "adam", "rmsprop"}

parser.add_argument(
    "--lamda", type=float, default=1
)  # recommended value: {0.001, 0.01, 0.1, 1.0}

parser.add_argument(
    "--sigma", type=float, default=100
)  # recommended value: {0.001, 0.01, 0.1, 1.0}

args = parser.parse_args()

# get raw dataset and build corresponding dataloader
root = "../../datasets/mnist/"
trainset = datasets.MNIST(
    root=root, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,)

testset = datasets.MNIST(
    root=root, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(
    testset, batch_size=len(testset), drop_last=False, shuffle=False
)

# setup
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = get_best_gpu()
else:
    device = torch.device("cpu")

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client
criterion = nn.CrossEntropyLoss()


if args.partition == "noniid":
    data_indices = noniid_slicing(
        dataset=trainset, num_clients=total_client_num, num_shards=400
    )
else:
    data_indices = random_slicing(dataset=trainset, num_clients=total_client_num)

optimizer_dict = dict(sgd=optim.SGD, adam=optim.Adam, rmsprop=optim.RMSprop)
optimizer = optimizer_dict[args.optimizer]

# initialize training dataloaders of each client
to_select = [i for i in range(args.total_client)]
trainloader_list = [
    DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        sampler=SubsetSampler(indices=data_indices[i]),
    )
    for i in to_select
]    

model = [CNN_MNIST().to(device) for _ in to_select]
flatten = lambda model: torch.cat([param.view(-1) for param in model.parameters()])
e = lambda x: math.exp(-x/args.sigma)/args.sigma

acc_list = [ 0 for _ in range(args.total_client)]

for i in range(args.total_client):
    if not load_model(i, model):
        model_parameters = FedAmpTrainer(
            model=model[i],
            data_loader=trainloader_list[i],
            epochs=args.epochs,
            optimizer=optimizer(model[i].parameters(), lr=args.lr),
            criterion=criterion,
            args=args,
        ).train(SerializationTool.serialize_model(model[i]))
        SerializationTool.deserialize_model(model[i], model_parameters)

        loss, acc = evaluate(model[i], criterion, test_loader)
        acc_list[i] = acc
        print(f"Epoch: {i}    loss: {loss:.4f}    accuracy: {acc:.2f}")
        save_model(i, model)
    else:
        model[i] = load_model(i, model).to(device)

# for i in range(args.total_client):
#     print(i)
#     model_parameters = FedAmpTrainer(
#         model=model[i],
#         data_loader=trainloader_list[i],
#         epochs=args.epochs,
#         optimizer=optimizer(model[i].parameters(), lr=args.lr),
#         criterion=criterion,
#         args=args,
#     ).train(SerializationTool.serialize_model(model[i]), SerializationTool.serialize_model(model[i]))
#     SerializationTool.deserialize_model(model[i], model_parameters)

#     loss, acc = evaluate(model[i], criterion, test_loader)
#     acc_list[i] = acc
#     print(f"Epoch: {i}    loss: {loss:.4f}    accuracy: {acc:.2f}")
# train
for i in range(args.round):
    selections = random.sample(to_select, num_per_round)
    params_list = []
    client_epoch = [args.epochs] * len(selections)
    cloud_model = deepcopy(model)
    for c_m in cloud_model:
        c_m.to(device)

    # local train
    for index in range(len(selections)):
        model[selections[index]] = load_model(selections[index], model).to(device)
        for param in cloud_model[selections[index]].parameters():
            param.data.zero_()
        coef = torch.zeros(len(selections)).to(device)
        for j in range(len(selections)):
            if selections[j] != selections[index]:
                wi = flatten(model[selections[index]]).to(device)
                wj = flatten(model[selections[j]]).to(device)
                diff = (wi - wj).view(-1)
                # print(torch.dot(diff, diff))
                coef[j] = args.alphaK * e(torch.dot(diff, diff))
            else:
                coef[j] = 0
        coef[index] = 1 - torch.sum(coef)
        for j in range(len(selections)):
            for cloud, local in zip(cloud_model[selections[index]].parameters(), model[selections[j]].parameters()):
                cloud.data += coef[j] * local.data
        
    for index in range(len(selections)):
        print(i, index, selections[index])
        model_parameters = FedAmpTrainer(
            model=model[selections[index]],
            data_loader=trainloader_list[selections[index]],
            epochs=client_epoch[index],
            optimizer=optimizer(model[selections[index]].parameters(), lr=args.lr),
            criterion=criterion,
            args=args,
        ).train(SerializationTool.serialize_model(cloud_model[selections[index]]))
        SerializationTool.deserialize_model(model[selections[index]], model_parameters)

        loss, acc = evaluate(model[selections[index]], criterion, test_loader)
        acc_list[selections[index]] = acc
        print(f"Epoch: {i}    loss: {loss:.4f}    accuracy: {acc:.2f}")
        save_model(selections[index], model)
        

    # update global model
    # aggregated_params = aggregator(params_list)
    # SerializationTool.deserialize_model(model, aggregated_params)

    # evaluate
    print(acc_list)
    print(max(acc_list))
    # loss, acc = evaluate(model, criterion, test_loader)
    # print(f"Epoch: {i}    loss: {loss:.4f}    accuracy: {acc:.2f}")


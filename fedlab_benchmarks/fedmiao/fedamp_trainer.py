# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 16:44:47
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 16:46:10

import torch
from copy import deepcopy

from fedlab.core.client import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import Logger
from tqdm import tqdm




class FedAmpTrainer(ClientTrainer):
    """FedProxTrainer. 

    Details of FedProx are available in paper: https://arxiv.org/abs/1812.06127

    Args:
        model (torch.nn.Module): PyTorch model.
        data_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        epochs (int): the number of local epoch.
        optimizer (torch.optim.Optimizer, optional): optimizer for this client's model.
        criterion (torch.nn.Loss, optional): loss function used in local training process.
        cuda (bool, optional): use GPUs or not. Default: ``True``.
        logger (Logger, optional): :object of :class:`Logger`.
        mu (float): hyper-parameter of FedProx.
    """

    def __init__(
        self,
        model,
        data_loader,
        epochs,
        optimizer,
        criterion,
        args,
        cuda=True,
        logger=Logger(),
    ):
        super().__init__(model, cuda)

        # self.model = model
        self._data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self._LOGGER = Logger() if logger is None else logger
        self.model_time = 0
        
        self.alphaK = args.alphaK
        self.lamda = args.lamda

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def local_process(self, payload):
        local_parameters, cloud_parameters = payload[0]
        self.train(local_parameters, cloud_parameters)


    def train(self, model_parameters, cloud_parameters):
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        u_cloud = deepcopy(self.model)
        SerializationTool.deserialize_model(u_cloud, cloud_parameters)
        SerializationTool.deserialize_model(self.model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self.model.train()
            for inputs, labels in tqdm(
                self._data_loader, desc="{}, Epoch {}".format(self._LOGGER.name, ep)
            ):
                if self.cuda:
                    inputs, labels = inputs.cuda(self.gpu), labels.cuda(self.gpu)

                outputs = self._model(inputs)
                l1 = self.criterion(outputs, labels)
                l2 = 0.0

                for w0, w in zip(u_cloud.parameters(), self.model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + self.lamda / (2 * self.alphaK) * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")

        return self.model_parameters

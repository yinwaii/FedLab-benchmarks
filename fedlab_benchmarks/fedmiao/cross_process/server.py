# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 17:39:06
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 17:39:18

import torch

import os
import fnmatch
import numpy as np
import sys

sys.path.append("../../../FedLab/")

from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.core.server.scale.manager import ScaleSynchronousManager
from fedlab.core.network import DistNetwork
from fedlab.core.communicator.package import Package
from fedlab.core.communicator.processor import PackageProcessor
from fedlab.utils.functional import AverageMeter, load_dict
from fedlab.utils.message_code import MessageCode
from fedlab.utils import SerializationTool, Aggregators, Logger

from config import local_grad_vector_file_pattern, clnt_params_file_pattern, \
    local_grad_vector_list_file_pattern, clnt_params_list_file_pattern
import models
from utils import load_local_grad_vector, load_clnt_params, evaluate

class AMPServerHandler:
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation.
        global_round (int): stop condition. Shut down FL system when global round is reached.
        sample_ratio (float): The result of ``sample_ratio * client_num`` is the number of clients for every FL round.
        cuda (bool): Use GPUs or not. Default: ``False``.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 global_round,
                 sample_ratio,
                 cuda=False,
                 logger=None):
        self.cuda = cuda
		self.models = 

        if cuda:
            # dynamic gpu acquire.
            self.gpu = get_best_gpu()
            self._model = model.cuda(self.gpu)
        else:
            self._model = model.cpu()

        self._LOGGER = Logger() if logger is None else logger

        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.client_num_in_total = 0
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []

        # stop condition
        self.global_round = global_round
        self.round = 0

    @property
    def downlink_package(self):
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``1`` to
        ``self.client_num_in_total + 1``."""
        selection = random.sample(range(self.client_num_in_total),
                                  self.client_num_per_round)
        return selection

    def _update_global_model(self, payload):
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round
        
        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(
                model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

class FedAvgServerHandler(SyncParameterServerHandler):
    def __init__(self, 
                 model,
                 test_loader,
                 weight_list=None,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 args=None):
        # get basic model
        # model = getattr(models, args['model_name'])(args['model_name'])
        super().__init__(model,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        self.test_loader = test_loader
        self.args = args
        self.weight_list = weight_list
        self.client_this_round = []
        self.acc_ = []
        self.loss_ = []

    def _update_global_model(self, model_parameters_list):
        self._LOGGER.info(
            "Model parameters aggregation, number of aggregation elements {}".
                format(len(model_parameters_list)))
        # use aggregator
        curr_weight_sum = sum([self.weight_list[cid] for cid in self.client_this_round])
        serialized_parameters = Aggregators.fedavg_aggregate(
            model_parameters_list) * len(self.client_this_round) / curr_weight_sum
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # evaluate on test set
        test_loss, test_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(),
                                       self.test_loader)
        self.acc_.append(test_acc)
        self.loss_.append(test_loss)
        self.write_file()

        # reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False

    def add_model(self, sender_rank, model_parameters):
        self.client_buffer_cache.append(model_parameters.clone())
        self.cache_cnt += 1

        # cache is full
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(self.client_buffer_cache)
            self.round += 1
            return True
        else:
            return False

    def write_file(self):
        file_name = os.path.join(self.args['out_dir'],
                                 f"{self.args['model_name']}_{self.args['partition']}_{self.args['dataset']}.txt")
        record = open(file_name, "w")

        record.write(str(self.args) + "\n")
        record.write(f"acc:" + str(self.acc_) + "\n")
        record.write(f"loss:" + str(self.loss_) + "\n")
        record.close()
import math
import threading
import torch
import numpy as np

from copy import deepcopy
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.server.handler import SyncParameterServerHandler
from fedlab.utils import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu
from fedlab.utils.message_code import MessageCode

class FedAmpManager(SynchronousServerManager):

    def __init__(self, network, handler, logger=None):
        super().__init__(network, handler, logger)

    def setup(self):
        super().setup()
        self._LOGGER.info(f"SETUP: {self._handler.client_num_in_total}")
        self._handler.cloud_model = [deepcopy(self._handler._model).to(self._handler.device) for _ in range(self._handler.client_num_in_total)]
        self._handler._model = [deepcopy(self._handler._model).to(self._handler.device) for _ in range(self._handler.client_num_in_total)]
        self._acc = [0 for _ in range(self._handler.client_num_in_total)]
        self._loss = [np.nan for _ in range(self._handler.client_num_in_total)]

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The id of clients are obtained from :meth:`handler.sample_clients`. And their communication ranks are are obtained via coordinator.
        """
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package(rank, clients_this_round)
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.ParameterUpdate,
                               dst=rank)

    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server backend.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ParameterServerBackendHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        while self._handler.if_stop is not True:
            self._LOGGER.info("Activating clients")
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler._update_global_model(sender_rank, payload):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

class FedAmpHandler(SyncParameterServerHandler):
    def __init__(self, 
                 model,
                 test_loader,
                 global_round=5,
                 cuda=False,
                 sample_ratio=1.0,
                 logger=None,
                 args=None):
        # get basic model
        # model = getattr(models, args['model_name'])(args['model_name'])
        assert(not (cuda and (not torch.cuda.is_available())))
        super().__init__(model,
                         global_round=global_round,
                         cuda=cuda,
                         sample_ratio=sample_ratio,
                         logger=logger)

        if cuda and torch.cuda.is_available():
            self.device = get_best_gpu()
        else:
            self.device = torch.device("cpu")

        self.test_loader = test_loader
        self.alphaK = args.alphaK
        self.sigma = args.sigma
        self._acc = []
        self._loss = []
        self.cache_cnt = 0

    def downlink_package(self, rank, clients_this_round):
        self._LOGGER.info(f"DOWNLINK: {rank}")
        rank -= 1
        flatten = lambda model: torch.cat([param.view(-1) for param in model.parameters()])
        e = lambda x: math.exp(-x/self.sigma)/self.sigma
        for param in self.cloud_model[rank].parameters():
            param.data.zero_()
        coef = torch.zeros(self.client_num_per_round).to(self.device)
        rank_index = -1
        for j in range(self.client_num_per_round):
            if clients_this_round[j] != rank:
                wi = flatten(self._model[rank]).to(self.device)
                wj = flatten(self.model[clients_this_round[j]]).to(self.device)
                diff = (wi - wj).view(-1)
                # print(torch.dot(diff, diff))
                coef[j] = self.alphaK * e(torch.dot(diff, diff))
            else:
                coef[j] = 0
                rank_index = j
        coef[rank_index] = 1 - torch.sum(coef)
        for j in range(self.client_num_per_round):
            for cloud, local in zip(self.cloud_model[rank].parameters(), self._model[clients_this_round[j]].parameters()):
                cloud.data += coef[j] * local.data
        return [SerializationTool.serialize_model(self.cloud_model[rank])]

    def _update_global_model(self, sender_rank, model_parameters_list):
        sender_rank -= 1
        self._LOGGER.info(
            "Model parameters aggregation, number of aggregation elements {}".
                format(len(model_parameters_list)))
        SerializationTool.deserialize_model(self._model[sender_rank],  model_parameters_list[0])
        
        # evaluate on test set
        test_loss, test_acc = evaluate(self._model[sender_rank], torch.nn.CrossEntropyLoss(),
                                       self.test_loader)
        print(f"Epoch: {sender_rank}    loss: {test_loss:.4f}    accuracy: {test_acc:.2f}")
        self._acc[sender_rank] = test_acc
        self._loss[sender_rank] = test_loss
        self.write_file()

        self.cache_cnt += 1
        return self.cache_cnt >= self.client_num_per_round
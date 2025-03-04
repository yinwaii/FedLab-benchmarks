# -*- coding: utf-8 -*-
# @Author: yinwai
# @Date:   2022-06-23 16:46:19
# @Last Modified by:   yinwai
# @Last Modified time: 2022-06-23 16:46:25

import os, torch
def save_model(index, model):
	model_path = os.path.join("models")
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	model_path = os.path.join(model_path, f"_{index}" + ".pt")
	torch.save(model[index], model_path)

def load_model(index, model):
	model_path = os.path.join("models")
	model_path = os.path.join(model_path, f"_{index}" + ".pt")
	if os.path.exists(model_path):
		return torch.load(model_path)
	else:
		return False
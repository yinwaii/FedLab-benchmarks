import os, torch, sys, math
from fedlab.utils.functional import get_best_gpu
sys.path.append("../")
from spacy import load
def save_model(index, model):
    model_path = os.path.join("models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, f"_{index}" + ".pt")
    torch.save(model[index], model_path)

def load_model(index):
    model_path = os.path.join("models")
    model_path = os.path.join(model_path, f"_{index}" + ".pt")
    if os.path.exists(model_path):
        return torch.load(model_path)
    else:
        return False

if torch.cuda.is_available():
    device = get_best_gpu()
else:
    device = torch.device("cpu")

from models.cnn import CNN_MNIST
flatten = lambda model: torch.cat([param.view(-1) for param in model.parameters()])
e = lambda x: math.exp(-x/100)/100

model = [load_model(i).to(device) for i in range(20)]
coef = torch.zeros((20,20))
print(model)
for i in range(20):
    for j in range(20):
        if j != i:
            wi = flatten(model[i]).to(device)
            wj = flatten(model[j]).to(device)
            diff = (wi - wj).view(-1)
            # print(torch.dot(diff, diff))
            coef[i][j] = 10000 * e(torch.dot(diff, diff))
        else:
            coef[i][j] = 0
# for index in range(20):
#     coef[index][index] = 1 - torch.sum(coef[index])
print(coef)
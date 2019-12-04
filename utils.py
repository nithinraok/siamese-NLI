import torch
import os

ROOT=os.getcwd()

def savemodel(model,dir='',device="cpu"):
    ROOT_PATH=os.path.join(ROOT,'saved_models_rep/',dir)
    model.to(device)
    if not os.path.exists(ROOT_PATH):
        os.makedirs(ROOT_PATH)
    PATH=os.path.join(ROOT_PATH,'saved_model.pth')
    torch.save(model,PATH)
    print("saved model")
	# model.to(device)

def loadmodel(dir='',device="cpu"):
    ROOT_PATH=os.path.join(ROOT,'saved_models/',dir)
    PATH=os.path.join(ROOT_PATH,'saved_model.pth')
    model=torch.load(PATH)
    return  model

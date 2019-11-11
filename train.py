# %matplotlib inline
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import savemodel,loadmodel
# import os
# import argparse
# from tensorboard_logger import configure, log_value

torch.manual_seed(42)
np.random.seed(42)

from SNLI_data import SNLI_dataset
from net import Siamese

def fit(model,loader,criteria,optimizer,device):
    tr_loss=0
    tr_acc=0
    count=0
    for batch in loader:
        text,hyp,label=[_.to(device) for _ in batch]
        
        logits = model(text,hyp)

        loss = criteria(logits,label)

        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        
        correct = ((logits.sigmoid()>0.5)==label)
        tr_acc += correct.type(torch.FloatTensor).mean().item()
        tr_loss += loss.item()
        count+=1
    
    return tr_loss/count,tr_acc/count

def valid(model,loader,criteria,optimizer,device):
    val_loss=0
    val_acc=0
    count=0
    for batch in loader:
        text,hyp,label=[_.to(device) for _ in batch]
        
        logits = model(text,hyp)

        loss = criteria(logits,label)

        correct = ((logits.sigmoid()>0.5)==label)
        val_acc += correct.type(torch.FloatTensor).mean().item()
        val_loss += loss.item()
        count+=1
    
    return val_loss/count,val_acc/count

batch_size=512
train_ds=SNLI_dataset('train')
train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=6)

valid_ds=SNLI_dataset('dev')
valid_dl=DataLoader(valid_ds,batch_size=batch_size,shuffle=False,num_workers=6)

test_ds=SNLI_dataset('test')
test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=6)

device='cpu'
model = Siamese(4096,1024,32)
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0.0)
criteria=torch.nn.BCEWithLogitsLoss()
model.to(device)

id=1
n_epochs=100
min_loss=float('Inf')
for epoch in range(1,n_epochs+1):
    model.train()
    tr_loss,tr_acc=fit(model,train_dl,criteria,optimizer,device)
    model.eval()
    val_loss,val_acc=valid(model,valid_dl,criteria,optimizer,device)
    test_loss,test_acc=valid(model,test_dl,criteria,optimizer,device)
    if val_loss<min_loss:
        savemodel(model,dir='siamese')
        min_loss=val_loss
    if epoch%id==0:
        print("tr_loss {:.3f} acc {:.3f} valid_loss {:.3f} acc {:.3f} test_loss {:.3f} acc {:.3f}".format(tr_loss,tr_acc,val_loss,val_acc,test_loss,test_acc))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import savemodel,loadmodel
# import os
import argparse
from tensorboard_logger import configure, log_value

torch.manual_seed(42)
np.random.seed(42)

from SNLI_data import SNLI_dataset,SNLI_Transformer
from net import Siamese,Siamese_LSTM

def fit(model,loader,criteria,optimizer,device):
    tr_loss=0
    tr_acc=0
    count=0
    for batch in loader:
        text,hyp,label,text_len,hyp_len=[_.to(device) for _ in batch]
        text=torch.nn.utils.rnn.pack_padded_sequence(text, text_len, batch_first=True, enforce_sorted=False)
        hyp=torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_len, batch_first=True, enforce_sorted=False)

        logits = model(text,hyp)

        loss = criteria(logits,label)

        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        
        pred = (logits.sigmoid()>0.5)
        correct = (pred==label)
        
        tr_acc += correct.type(torch.FloatTensor).mean().item()
        tr_loss += loss.item()
        count+=1
        if count>100:
            break
        if count%20==0:
            print("Batch {}/100 loss: {:.3f} acc: {:.3f} ".format(count,tr_loss/count,tr_acc/count))
    
    return tr_loss/count,tr_acc/count

def valid(model,loader,criteria,optimizer,device):
    val_loss=0
    val_acc=0
    count=0
    for batch in loader:
        text,hyp,label,text_len,hyp_len=[_.to(device) for _ in batch]
        text=torch.nn.utils.rnn.pack_padded_sequence(text, text_len, batch_first=True, enforce_sorted=False)
        hyp=torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_len, batch_first=True, enforce_sorted=False)
        
        logits = model(text,hyp)

        loss = criteria(logits,label)

        correct = ((logits.sigmoid()>0.5)==label)
        val_acc += correct.type(torch.FloatTensor).mean().item()
        val_loss += loss.item()
        count+=1
        
    return val_loss/count,val_acc/count

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",default=512, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    args = parser.parse_args()

    batch_size=args.batch_size
    n_epochs=args.epochs
    train_ds=SNLI_Transformer('train')
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=6)

    valid_ds=SNLI_Transformer('dev')
    valid_dl=DataLoader(valid_ds,batch_size=batch_size,shuffle=False,num_workers=6)

    test_ds=SNLI_Transformer('test')
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=6)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    # model = Siamese(4096,1024,32)
    model = Siamese_LSTM(768,256,batch_size=batch_size,num_layers=1,out_dim=32,device=device)
    print(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=0.9)
    criteria=torch.nn.BCEWithLogitsLoss()
    model.to(device)

    name = "epoch_{}_batch_{}_FNN_siamese_transformer".format(n_epochs,batch_size)
    configure("logs/{}".format(name))

    id=1
    
    min_loss=float('Inf')
    for epoch in range(1,n_epochs+1):
        model.train()
        tr_loss,tr_acc=fit(model,train_dl,criteria,optimizer,device)
        model.eval()
        val_loss,val_acc=valid(model,valid_dl,criteria,optimizer,device)
        test_loss,test_acc=valid(model,test_dl,criteria,optimizer,device)

        log_value('Loss/train',tr_loss,epoch)
        log_value('Accuracy/train',tr_acc,epoch)
        log_value('Loss/valid',val_loss,epoch)
        log_value('Accuracy/valid',val_acc,epoch)
        log_value('Loss/test',test_loss,epoch)
        log_value('Accuracy/test',test_acc,epoch)
        if val_loss<min_loss:
            savemodel(model,dir='siamese')
            min_loss=val_loss
        if epoch%id==0:
            print("tr_loss {:.3f} acc {:.3f} valid_loss {:.3f} acc {:.3f} test_loss {:.3f} acc {:.3f}".format(tr_loss,tr_acc,val_loss,val_acc,test_loss,test_acc))

import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import os

ROOT=os.getcwd()

class SNLI_dataset(Dataset):
    def __init__(self,split):
        super().__init__()

        PATH1 = osp.join(ROOT,'data',split,'contradiction.npy')
        PATH2 = osp.join(ROOT,'data',split,'entailment.npy')

        self.contra = np.load(PATH1,mmap_mode='r')
        self.entail = np.load(PATH2,mmap_mode='r')
        

    def __len__(self):
        return (len(self.contra)+len(self.entail))//2
    
    def __getitem__(self,val):
        index=val*2
        if index<len(self.contra):
            text = torch.from_numpy(self.contra[index])
            hyp = torch.from_numpy(self.contra[index+1])
            y = torch.tensor(1,dtype=torch.float32)
            return text,hyp,y
        else:
            index=index-len(self.contra)
            text = torch.from_numpy(self.entail[index])
            hyp = torch.from_numpy(self.entail[index+1])
            y = torch.tensor(0,dtype=torch.float32)
            return text,hyp,y
    


class SNLI_Transformer(Dataset):
    def __init__(self,split):
        super().__init__()

        PATH1 = osp.join(ROOT,'data',split,'contradiction.txt')
        PATH2 = osp.join(ROOT,'data',split,'entailment.txt')

        with open(PATH1,'r') as f:
            self.contra = f.readlines()
        
        with open(PATH2,'r') as f:
            self.entail = f.readlines()
        

    def __len__(self):
        return (len(self.contra)+len(self.entail))//2
    
    def __getitem__(self,val):
        index=val*2
        if index<len(self.contra):
            text = self.contra[index].strip()
            hyp = self.contra[index+1].strip()
            y = torch.tensor(1,dtype=torch.float32)
            return text,hyp,y
        else:
            index=index-len(self.contra)
            text = torch.from_numpy(self.entail[index])
            hyp = torch.from_numpy(self.entail[index+1])
            y = torch.tensor(0,dtype=torch.float32)
            return text,hyp,y
      
    

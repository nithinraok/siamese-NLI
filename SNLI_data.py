import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import os

    
class Data():
    pass

ROOT=os.getcwd()
BERT = Data()
BERT.test = 1
BERT.dev = 1
BERT.train = 2

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
        
        self.contra=Data()
        self.entail=Data()
        self.contra.len=0
        self.entail.len=0
        for i in range(getattr(BERT,split)):
            f_name = "{}/{}/{}/bert_{}_contradiction.npy".format(ROOT,'data_8k',split,i)
            attr = "attr_{}".format(i)
            setattr(self.contra,attr,np.load(f_name,mmap_mode='r'))
            self.contra.len+=len(getattr(self.contra,attr))

            f_name = "{}/{}/{}/bert_{}_entailment.npy".format(ROOT,'data_8k',split,i)
            setattr(self.entail,attr,np.load(f_name,mmap_mode='r'))
            self.entail.len+=len(getattr(self.entail,attr))
        
        for i in range(getattr(BERT,split)):
            f_name = "{}/{}/{}/sent_bert_{}_contradiction.npy".format(ROOT,'data_8k',split,i)
            attr = "sent_{}".format(i)
            setattr(self.contra,attr,np.load(f_name,mmap_mode='r'))


            f_name = "{}/{}/{}/sent_bert_{}_entailment.npy".format(ROOT,'data_8k',split,i)
            setattr(self.entail,attr,np.load(f_name,mmap_mode='r'))

        

    def __len__(self):
        return ((self.contra.len)+(self.entail.len))//2
    
    def get_emb(self,object,index):
        rem = index//50000
        attr = "attr_{}".format(rem)
        index = index%50000
        return torch.from_numpy(getattr(object,attr)[index])

    def get_length(self,object,index):
        rem = index//50000
        attr = "sent_{}".format(rem)
        index = index%50000
        return torch.from_numpy(np.array(getattr(object,attr)[index]))

    
    def __getitem__(self,val):
        index=val*2
        if index<(self.contra.len):
            text = self.get_emb(self.contra,index)
            hyp = self.get_emb(self.contra,index+1)
            text_len = self.get_length(self.contra,index)
            hyp_len = self.get_length(self.contra,index+1)
            y = torch.tensor(1,dtype=torch.float32)
            text_len = torch.clamp(text_len,1,32)
            hyp_len = torch.clamp(hyp_len,1,32)
            return text,hyp,y,text_len,hyp_len
        else:
            index=index-(self.contra.len)
            text = self.get_emb(self.entail,index)
            hyp = self.get_emb(self.entail,index+1)
            text_len = self.get_length(self.entail,index)
            hyp_len = self.get_length(self.entail,index+1)
            y = torch.tensor(0,dtype=torch.float32)
            text_len = torch.clamp(text_len,1,32)
            hyp_len = torch.clamp(hyp_len,1,32)
            return text,hyp,y,text_len,hyp_len
      
    

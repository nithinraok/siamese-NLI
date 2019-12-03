import torch.nn as nn
import torch

class Siamese(nn.Module):
	def __init__(self,in_dim,hid_dim,out_dim):
		super(Siamese,self).__init__()
		self.encoder=nn.Sequential(
		nn.Linear(in_dim,hid_dim),
		nn.ReLU(),
		nn.Linear(hid_dim,hid_dim//2),
		nn.BatchNorm1d(hid_dim//2),
		nn.ReLU(),
		nn.Dropout(p=0.2),
		nn.Linear(hid_dim//2,hid_dim//3),
		nn.BatchNorm1d(hid_dim//3),
		nn.ReLU(),
		nn.Dropout(p=0.2),
		nn.Linear(hid_dim//3,out_dim)
		)
		# self.linear=nn.Linear(out_dim,1,bias=False)
		S=torch.randn((out_dim,out_dim),requires_grad=True)
		self.S = nn.Parameter(S,requires_grad=True)

		b=torch.randn(1,requires_grad=True)
		self.b = nn.Parameter(b,requires_grad=True)


	def forward(self,inp1,inp2):
		x=self.encoder(inp1)
		y=self.encoder(inp2)
		# out=torch.pow((out1-out2),2)
		out=torch.matmul(x,y.T).diag() - torch.matmul(torch.matmul(x,self.S),x.T).diag()-torch.matmul(torch.matmul(y,self.S),y.T).diag()+self.b
		return out

	def forward_once(self,inp):
		x=self.encoder(inp)
		return x


class Siamese_LSTM(nn.Module):
	def __init__(self,inp_dim,hid_dim,batch_size=512,num_layers=2,out_dim=32,device='cpu'):
		super().__init__()
		self.input_dim = inp_dim
		self.hidden_dim = hid_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.device= device

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)
		self.linear = nn.Linear(self.hidden_dim,out_dim)
		S=torch.randn((out_dim,out_dim),requires_grad=True)
		self.S = nn.Parameter(S,requires_grad=True)

		b=torch.randn(1,requires_grad=True)
		self.b = nn.Parameter(b,requires_grad=True)

	def init_hidden(self):
        
		return (torch.zeros(self.num_layers, self.batch_size,self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers,self.batch_size, self.hidden_dim).to(self.device))

	def forward(self, x,y):
		x = self.forward_once(x)
		y = self.forward_once(y)
		
		out=torch.matmul(x,y.T).diag() - torch.matmul(torch.matmul(x,self.S),x.T).diag()-torch.matmul(torch.matmul(y,self.S),y.T).diag()+self.b
		return out.view(-1)
	
	def forward_once(self,x):
		hidden = self.init_hidden()
		self.lstm.flatten_parameters()
		lstm_out, self.hidden = self.lstm(x.float(),hidden)
		unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
		out = unpacked[:,-1]
		out = self.linear(out)
		return out
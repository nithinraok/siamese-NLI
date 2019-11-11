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
import torch 


mean = torch.randn((32,100))
std = torch.ones((32,100))

a = torch.distributions.Normal(mean,std)

y = torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=std), 1)
print(y.rsample().shape)

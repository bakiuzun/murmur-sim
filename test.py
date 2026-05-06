import torch 


a = torch.randn((3,1)) * 3 


a = torch.clamp(a,1.0)

print(a)
import torch

x = torch.empty(5, 3)
x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5.5, 3])
x = x.new_ones(5,3,dtype=torch.double)
print(x)
x= torch.randn_like(x,dtype=torch.float)
print(x)
print(x.size())

if torch.cuda. is_available():
    device = torch.device("cuda")
    y= torch.ones_like(x,device=device)
    x = x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))
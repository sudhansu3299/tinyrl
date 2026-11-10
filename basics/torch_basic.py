import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = 3 * y

w = 4 * x

print(x.requires_grad)
print(y.grad_fn)
print(z.grad_fn)

z.backward(retain_graph=True)
print(x.grad)

d = x.grad.item()
print(d)
w.backward()
print(x.grad)
print(x.requires_grad)

print("END")

# import torch
#
# x = torch.tensor(3.0, requires_grad=True)
# y = x * 2
# z = y**2
#
# z.backward()
#
# print(x.grad)
# print(y.grad)
# print(y.is_leaf)

optimizer = torch.optim.SGD([x], lr=0.01)

optimizer.zero_grad(set_to_none=True)
z.backward()
optimizer.step()

print(x)
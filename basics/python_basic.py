#
# a = [1, 50, 3, 3] #list
# b = {1, 2, "3", "3", 3, 3} #set
# c = {1: "a", 2: "b", 3: "c"} #dict
# d = (1, 2, 3) # tuple
#
# actions = ["left", "right", "up", "down"]
# states = ["s1", "s2", "s3"]
#
# # for i, action in enumerate(actions, start=2):
# #     print(i, action)
# #
# # for i in range(len(actions)):
# #     print(i, actions[i])
# #
# # for state, action in zip(states, actions):
# #     print(state, action)
#
# x = a.copy()
# print(x)
#
# x.append(4)
# print(x)
# print(a)
#
# a[0] = 100
# print(x)
# a.sort(reverse=True)
# print(a)

import torch
import torch.nn as nn

torch.manual_seed(42)

states = torch.linspace(-1, 1, 100).unsqueeze(-1)

# Synthetic target values
returns = 1 - states**2

layer = torch.nn.Linear(1, 16)

output = layer(states)

model = nn.Sequential(
    nn.Linear(1,8),
    nn.Tanh(),
    nn.Linear(8, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fn = nn.MSELoss()

for epoch in range(10000):
    optimizer.zero_grad()

    y_pred = model(states)
    loss = loss_fn(y_pred, returns)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    predictions = model(states)

    print("Final predictions:\n", predictions)
    print("Final loss:", torch.mean((predictions - returns) ** 2).item())




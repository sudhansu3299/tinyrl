import torch
import torch.nn as nn

x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]]) #4x2

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]]) #4x1

torch.manual_seed(5)

#1st hidden layer, 4 neurons; 2nd one's output layer, 1 neuron
model = nn.Sequential(
    nn.Linear(2,4),
    nn.Sigmoid(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

loss_fn = nn.BCELoss()

for epoch in range(10000):
    optimizer.zero_grad() #make prev grads zero

    #1. Forward pass
    y_hat = model(x)

    # 2. Calculate MSE loss
    loss = loss_fn(y_hat, y)#torch.mean((y_hat - y) ** 2)

    #3. Backpropagation +update
    loss.backward() #calc. the gradients
    optimizer.step() #make the update

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    predictions = model(x)

    print("Final predictions:\n", predictions)
    print("Final loss:", torch.mean((predictions - y) ** 2).item())

# for name, param in model.named_parameters():
#     print(name, param)
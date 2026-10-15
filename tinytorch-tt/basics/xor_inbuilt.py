import torch

x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]]) #4x2

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]]) #4x1

torch.manual_seed(42)

#1st hidden layer, 2 neurons; 2nd hidden layer, 1 neuron
w1 = torch.randn(2,2, requires_grad=True)
w2 = torch.randn(2,1, requires_grad=True)

b1 = torch.zeros(2, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

alpha = 0.5

for epoch in range(10000):
    # 1. Forward pass
    h = torch.sigmoid(x @ w1 + b1)
    y_hat = torch.sigmoid(h @ w2 + b2)

    # 2. Calculate loss
    loss = torch.mean((y_hat - y) ** 2)

    # 3. Clear gradients from previous iteration
    for param in [w1, b1, w2, b2]:
        if param.grad is not None:
            param.grad.zero_()

    # 4. Automatic backpropagation
    loss.backward()

    # 5. Update parameters
    with torch.no_grad():
        for param in [w1, b1, w2, b2]:
            param -= alpha * param.grad

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    h = torch.sigmoid(x @ w1 + b1)
    predictions = torch.sigmoid(h @ w2 + b2)

    print("Final predictions:\n", predictions)
    print("Final loss:", torch.mean((predictions - y) ** 2).item())
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
    h = torch.sigmoid(x@w1 +b1)
    z = h@w2 + b2

    y_hat2 = torch.sigmoid(z)

    loss = torch.mean((y_hat2 - y) ** 2) #MSE LOSS

    #Implement backprop
    #Learning Gradient => dl/dw = dl/dy_hat * dy_hat/dz * dz/dw

    #Step1: dl/dy_hat
    dl_dy_hat = (2/ y.numel()) * (y_hat2 - y)

    #Step2: dy_hat/dz
    dy_hat_dz = y_hat2 * (1 - y_hat2)
    dl_dz2 = dl_dy_hat * dy_hat_dz

    # Step 3: output-layer gradients
    dl_dw2 = h.T @ dl_dz2
    dl_db2 = dl_dz2.sum(dim=0)

    # Step 4: propagate gradient into hidden layer
    dl_dh = dl_dz2 @ w2.T
    dl_dz1 = dl_dh * h * (1 - h)

    # Step 5: hidden-layer gradients
    dl_dw1 = x.T @ dl_dz1
    dl_db1 = dl_dz1.sum(dim=0)

    # Step 6: update all parameters
    with torch.no_grad():
        w2 -= alpha * dl_dw2
        b2 -= alpha * dl_db2
        w1 -= alpha * dl_dw1
        b1 -= alpha * dl_db1

    if epoch % 1000 ==0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    h = torch.sigmoid(x @ w1 + b1)
    predictions = torch.sigmoid(h @ w2 + b2)

    print("Final predictions:\n", predictions)
    print("Final loss:", torch.mean((predictions - y) ** 2).item())

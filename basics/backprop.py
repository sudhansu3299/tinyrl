import torch

x = torch.tensor([2.0, 3.0])

y = torch.tensor([1.0, 1.5])


#2 neuron hidden layer
w2 = torch.randn(2,2, requires_grad=True)
b2 = torch.zeros(2, requires_grad=True)

alpha = 0.1

for epoch in range(100):
    z2 = x@w2 + b2
    # y_hat2 = torch.sigmoid(z2)
    y_hat2 = z2

    loss = torch.mean((y_hat2 - y) ** 2) #MSE LOSS

    if w2.grad is not None:
        w2.grad.zero_()
        b2.grad.zero_()
    
    #Implement backprop
    #Learning Gradient => dl/dw = dl/dy_hat * dy_hat/dz * dz/dw
    
    #Step1: dl/dy_hat
    dl_dy_hat = (2/ y.numel()) * (y_hat2 - y)
    
    #Step2: dy_hat/dz
    # dy_hat_dz = y_hat2 * (1 - y_hat2)

    #Step3: dz/dw
    # dz_dw = x
    
    #Step4: dl/db = dl/dy_hat
    dl_db = dl_dy_hat
    dl_dz = dl_db
    
    #Step5: dl/dw
    dl_dw = torch.outer(x, dl_dz) #produces, (m,) x (n,) = (m,n) shape
    
    with torch.no_grad(): #here the computations won't be part of the computational graph
        w2 -= alpha * dl_dw
        b2 -= alpha * dl_db

    if epoch % 10 ==0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Final Prediction: ", y_hat2.detach())
































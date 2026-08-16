import torch

def sigmoid(zx):

    logits = 1/(1 + torch.exp(-zx))

    return logits

def generate_logistic_data(dimensions = 1, noise_constant = 3):
    noise = noise_constant * torch.randn((128,))
    b = torch.tensor([5.0])
    if dimensions == 1:
        x = torch.randint(-3, 3, (128,1)).float()
        w = torch.tensor([3.0])
        zx = x @ w + b + noise
    elif dimensions == 2:
        x = torch.randint(-3, 3,(128,2)).float()
        w = torch.tensor([3.0,2.0])
        zx = x @ w + b + noise
    else:
        x = torch.randint(-3, 3,(128,3)).float()
        w = torch.tensor([3.0,2.0,1.0])
        zx = x @ w + b + noise

    logits = sigmoid(zx)
    y = torch.bernoulli(logits)
    return x, y 



def split_data(x,y, train_ratio = 0.75):
    permuatation = torch.randperm(x.shape[0])
    cutoff = int(x.shape[0] * train_ratio)

    train_indices = permuatation[:cutoff]
    test_indices = permuatation[cutoff:]
    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_test, y_test

def forward(x, w, b):
    zx = x @ w + b
    logits = sigmoid(zx)
    
    return logits 

def predict(logits, threshold = 0.5):
    
    return (logits >= threshold).float()


def back_prop(y_pred, y_real, x):

    dwx = (1 / len(y_pred)) * torch.matmul(x.T, (y_pred - y_real))
    dbx = torch.sum(1/len(y_pred) * (y_pred - y_real))

    return dwx, dbx

def bce_loss(y_pred, y_real, eps = 1e-5):
    p = torch.clamp(y_pred,eps, 1- eps)
    loss = - (y_real * torch.log(p) + (1-y_real) * torch.log(1 - p)).mean()
    return loss

def epochs(x, y_real, epochs, epsilon, learning_rate = 1e-3 ):
    w = torch.randn(x.shape[1])
    b = 0
    losses = []
    dw = []
    db = []
    for i in range(epochs):
        y_pred = forward(x, w, b)
        ls = bce_loss(y_pred, y_real)
        dwx, dbx = back_prop(y_pred, y_real, x)
        w -= learning_rate * dwx
        b -= learning_rate * dbx
        dw.append(dwx)
        db.append(dbx)
        if losses and abs(ls - losses[-1]) < epsilon:
            losses.append(ls)
            break
        losses.append(ls)

    return w, b, dw, db, losses
    



    







    




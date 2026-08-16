import torch
torch.manual_seed(42)

def generate_regression_data(dimensions = 1, noise_constant = 3):
    noise = noise_constant * torch.randn((128,))
    b = torch.tensor([5.0])
    if dimensions == 1:
        x = 10 * torch.rand((128,1)) 
        w = torch.tensor([3.0])
        y = x @ w + b + noise
    elif dimensions == 2:
        x = 10 * torch.rand((128,2))
        w = torch.tensor([3.0,2.0])
        y = x @ w + b + noise
    else:
        x = 10 * torch.rand((128,3)) 
        w = torch.tensor([3.0,2.0,1.0])
        y = x @ w + b + noise
    
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
    y = x @ w + b
    return y

def back_prop(y_pred, y_real, x):

    dwx = (1 / len(y_pred)) * torch.matmul(x.T, (y_pred - y_real))
    dbx = torch.sum(1/len(y_pred) * (y_pred - y_real))

    return dwx, dbx

def loss(y_pred, y_real):
    
    return torch.sum(1/(2 * len(y_pred)) * (y_pred - y_real) ** 2)

def epochs(x, y_real, epochs, epsilon, learning_rate = 1e-3 ):
    w = torch.randn(x.shape[1])
    b = 0
    losses = []
    dw = []
    db = []
    for i in range(epochs):
        y_pred = forward(x, w, b)
        ls = loss(y_pred, y_real)
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


def closed_formed(X_train, y_train, w, b):
    # X Shape (a,b)
    # Y Shape (a,1)

    # w shape (b,)
    # b shape (1,1)

    a = X_train.shape[0]

    ones = torch.ones( (a,1), dtype = X_train.dtype)

    X = torch.cat([X_train, ones], dim=1)

    theta = torch.linalg.pinv(X) @ y_train

    w = theta[:-1]
    b = theta[-1]

    return w,b 
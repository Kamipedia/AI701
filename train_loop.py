import os
import cv2
import numpy as np
import torch    
import random
import gc
from glob import glob
from tqdm import tqdm

def train_epoch(model, train_dl, optimizer, criterion, scheduler=None):
    model.train()
    loss_history = 0

    for x_batch, y_batch in tqdm(train_dl, leave=False):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss)
        loss_history += loss.item() * x_batch.size(0)

    return loss_history / len(train_dl.dataset)


@torch.no_grad()
def val_epoch(model, val_dl, criterion):
    model.eval()
    loss_history = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_dl, leave=False):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss_history += loss.item() * x_batch.size(0)
            predicted = torch.argmax(y_pred.data, 1)
            y_batch = torch.argmax(y_batch.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return loss_history / len(val_dl.dataset), correct / total


def train(model, 
          train_dl, 
          val_dl, 
          optimizer, 
          criterion, 
          scheduler=None, 
          epochs=50, 
          val_ep=10, 
          task="baseline", 
          tolerance=3, 
          tol_threshold=0.01, 
          early_stopping=True
          ):
    
    train_hist = []
    acc_hist = []
    val_loss = 0
    best_loss = np.inf

    for ep in range(epochs+1):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, scheduler)

        if (ep) % val_ep == 0:
            val_loss, val_acc = val_epoch(model, val_dl, criterion)
            print(f'Epoch {ep}: train loss {train_loss:.4}, val loss {val_loss:.4}')

            if abs(val_loss) < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_{task}.pth')

        else:
            print(f'Epoch {ep}: train loss {train_loss:.4}, best val loss {best_loss:.4}')

        train_hist.append((train_loss, val_loss))
        acc_hist.append(val_acc)
        if early_stopping and ep>tolerance*val_ep and val_loss-np.mean(np.array(train_hist)[-tolerance*val_ep:,1])>=tol_threshold:
                print('Early stopping')
                break
    
    optimizer.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

    print("Training finished, best val loss: ", f"{best_loss:.4}", "recovering best model")
    model.load_state_dict(torch.load(f'best_{task}.pth'))

    return train_hist, acc_hist


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


from typing import List
import tqdm

import torch
from torch.utils.data import DataLoader


# Direct prediction accuracy computation
def direct_prediction_accuracy(model : torch.nn.Module, loader : DataLoader, num_variables : int, masked_idxs : List[int]):
    device = model.device
    acc = torch.tensor(0.0).to(device)
    acc_last = torch.tensor(0.0).to(device)
    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        
        # Make prediction
        y_pred = model(x)

        # Remove masked variables
        y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_variables)]))[0]]
        y = y[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_variables)]))[0]]
        
        y_pred = y_pred.softmax(dim=-1)

        # Calculate accuracy
        acc += (y_pred.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()
        acc_last += (y_pred[:,-1,:].argmax(dim=-1) == y[:,-1,:].argmax(dim=-1)).float().mean()
    
    acc /= len(loader)
    acc_last /= len(loader)

    return acc, acc_last



def entropy(y : torch.Tensor, n : int = 2):
    y = y.clamp(min=1e-8, max=1-1e-8)
    return -torch.sum(y * torch.log(y), dim=-1) / torch.log(torch.tensor(n).float())

def mutual_information(model : torch.nn.Module, loader : DataLoader, num_variables : int, masked_idxs : List[int]):
    y_preds = []
    ys = []
    device = model.device

    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y = y.to(device)

        # Make prediction
        y_pred = model(x)

        # Remove masked variables + keep only last timestep
        y_pred = y_pred[:,-1,torch.where(~torch.tensor([i in masked_idxs for i in range(num_variables)]))[0]]
        y = y[:,-1,torch.where(~torch.tensor([i in masked_idxs for i in range(num_variables)]))[0]]

        y_pred = y_pred.softmax(dim=-1)

        # Save predictions
        y_preds.append(y_pred)
        ys.append(y)

    # Compute probabilities p(y_pred) and p(y)
    y_preds = torch.cat(y_preds, dim=0)
    ys = torch.cat(ys, dim=0)
    
    p_y_pred = torch.mean(y_preds, dim=0)
    p_y = torch.mean(ys, dim=0)

    # Calculate joint probability p(y_pred, y) = p(y_pred|y) * p(y). y_pred and y are not independent because they have a common cause x. we cannot buld a shared histogram because y_pred contains probabilities and y contains one-hot vectors.
    remaining_vars = num_variables - len(masked_idxs)
    y_uniques = torch.unique(ys, dim=0) # Select all unique values of y (/!\ unstated assumption: all values of y are present in the dataset)
    y_freq = torch.stack([torch.sum(torch.all(ys == y_value, dim=-1)) for y_value in y_uniques], dim=0) / ys.size(0) # Compute the frequency of each unique value of y: p(y)
    y_pred_given_y = torch.stack([torch.stack([v[1] for v in torch.stack([ys,y_preds],-2) if (v[0] == y).all()]).mean(0) for y in y_uniques]) # Compute the probability of y_pred given each unique value of y: p(y_pred|y)
    joint_p_y_y_pred = (y_freq.unsqueeze(1).repeat(1,y_pred_given_y.size(-1)) * y_pred_given_y).view((remaining_vars**2,))# Compute the joint probability of y_pred and y: p(y_pred, y) = p(y_pred|y) * p(y)

    # Calculate mutual information I(y_pred; y) = H(y_pred) + H(y) - H(y_pred, y)
    mi = entropy(p_y_pred, remaining_vars) + entropy(p_y, remaining_vars) - entropy(joint_p_y_y_pred, remaining_vars)

    return mi




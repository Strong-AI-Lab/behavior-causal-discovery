
import tqdm

import torch
import torch.nn.functional as F


# Direct prediction accuracy computation
def direct_prediction_accuracy(model, loader, num_var, masked_idxs):
    acc = torch.tensor(0.0)
    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        # Make prediction
        y_pred = model(x)

        # Remove masked variables
        y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        y = y[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

        y_pred = y_pred.softmax(dim=-1)

        # Calculate accuracy
        acc = acc * i / (i+1) + (y_pred.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() / (i+1)
    return acc



def entropy(y, n=2):
    y = y.clamp(min=1e-8, max=1-1e-8)
    return -torch.sum(y * torch.log(y), dim=-1) / torch.log(torch.tensor(n).float())

def mutual_information(model, loader, num_var, masked_idxs):
    y_preds = []
    ys = []

    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        # Make prediction
        y_pred = model(x)

        # Remove masked variables + keep only last timestep
        y_pred = y_pred[:,-1,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        y = y[:,-1,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

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
    y_uniques = torch.unique(ys, dim=0) # Select all unique values of y
    y_freq = torch.stack([torch.sum(torch.all(ys == y_value, dim=-1)) for y_value in y_uniques], dim=0) / ys.size(0) # Compute the frequency of each unique value of y: p(y)
    y_pred_given_y = torch.stack([torch.stack([v[1] for v in torch.stack([ys,y_preds],-2) if (v[0] == y).all()]).mean(0) for y in y_uniques]) # Compute the probability of y_pred given each unique value of y: p(y_pred|y)
    joint_p_y_y_pred = (y_freq.unsqueeze(1).repeat(1,y_pred_given_y.size(-1)) * y_pred_given_y).view(((num_var - len(masked_idxs))**2,))# Compute the joint probability of y_pred and y: p(y_pred, y) = p(y_pred|y) * p(y)

    # Calculate mutual information I(y_pred; y) = H(y_pred) + H(y) - H(y_pred, y)
    mi = entropy(p_y_pred, num_var) + entropy(p_y, num_var) - entropy(joint_p_y_y_pred, num_var)

    return mi




# Single individual Series generation (context variables are left unchanged)
def generate_series(model, dataset, num_var, masked_idxs):
    prev_ind = -1
    series = {}
    for i, (x, y, ind) in enumerate(tqdm.tqdm(dataset)):
        if prev_ind != ind:
            prev_ind = ind
        else:
            x_obs = x[:,torch.where(torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
            x = torch.cat((series[ind][-1][0], x_obs), dim=1)

        # Make prediction
        y_pred = model(x.unsqueeze(0))[0]

        # Remove masked variables
        y_pred = y_pred[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        y = y[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        x = x[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

        y_pred = F.gumbel_softmax(y_pred.log(), hard=True)
        y_pred = torch.cat((x[1:,:], y_pred[-1:,:]), dim=0)

        # Update series
        if ind not in series:
            series[ind] = []
        series[ind].append((y_pred, y))
    return series # ((tau, num_var) (tau, num_var))

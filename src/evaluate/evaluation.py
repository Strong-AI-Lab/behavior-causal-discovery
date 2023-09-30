
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

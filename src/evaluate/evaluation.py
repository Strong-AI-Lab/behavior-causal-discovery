
import tqdm
from collections import Counter

import torch
import torch.nn.functional as F


# Direct prediction accuracy computation
def direct_prediction_accuracy(model, loader, num_var, masked_idxs):
    acc = torch.tensor(0.0)
    acc_last = torch.tensor(0.0)
    device = model.device
    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        
        # Make prediction
        y_pred = model(x)

        # Remove masked variables
        y_pred = y_pred[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        y = y[:,:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

        y_pred = y_pred.softmax(dim=-1)

        # Calculate accuracy
        acc = acc * i / (i+1) + (y_pred.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() / (i+1)
        acc_last = acc_last * i / (i+1) + (y_pred[:,-1,:].argmax(dim=-1) == y[:,-1,:].argmax(dim=-1)).float().mean() / (i+1)
    return acc, acc_last



def entropy(y, n=2):
    y = y.clamp(min=1e-8, max=1-1e-8)
    return -torch.sum(y * torch.log(y), dim=-1) / torch.log(torch.tensor(n).float())

def mutual_information(model, loader, num_var, masked_idxs):
    y_preds = []
    ys = []
    device = model.device

    for i, (x, y, _) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y = y.to(device)

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
    remaining_vars = num_var - len(masked_idxs)
    y_uniques = torch.unique(ys, dim=0) # Select all unique values of y (/!\ unstated assumption: all values of y are present in the dataset)
    y_freq = torch.stack([torch.sum(torch.all(ys == y_value, dim=-1)) for y_value in y_uniques], dim=0) / ys.size(0) # Compute the frequency of each unique value of y: p(y)
    y_pred_given_y = torch.stack([torch.stack([v[1] for v in torch.stack([ys,y_preds],-2) if (v[0] == y).all()]).mean(0) for y in y_uniques]) # Compute the probability of y_pred given each unique value of y: p(y_pred|y)
    joint_p_y_y_pred = (y_freq.unsqueeze(1).repeat(1,y_pred_given_y.size(-1)) * y_pred_given_y).view((remaining_vars**2,))# Compute the joint probability of y_pred and y: p(y_pred, y) = p(y_pred|y) * p(y)

    # Calculate mutual information I(y_pred; y) = H(y_pred) + H(y) - H(y_pred, y)
    mi = entropy(p_y_pred, remaining_vars) + entropy(p_y, remaining_vars) - entropy(joint_p_y_y_pred, remaining_vars)

    return mi




# Single individual Series generation (context variables are left unchanged)
def generate_series(model, dataset, num_var, masked_idxs):
    prev_ind = -1
    series = {}
    device = model.device
    for x, y, ind in tqdm.tqdm(dataset):
        x = x.to(device)
        y = y.to(device)

        if prev_ind != ind:
            prev_ind = ind
        else:
            hist = series[ind][-1][0]
            hist = hist.to(device)
            x_obs = x[:,torch.where(torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
            x = torch.cat((hist, x_obs), dim=1)

        # Make prediction
        y_pred = model(x.unsqueeze(0))[0]

        # Remove masked variables
        y_pred = y_pred[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        y = y[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
        x = x[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

        y_pred = F.gumbel_softmax(y_pred.clamp(min=1e-8, max=1-1e-8).log(), hard=True)
        y_pred = torch.cat((x[1:,:], y_pred[-1:,:]), dim=0)

        y_pred = y_pred.cpu()
        y = y.cpu()

        # Update series
        if ind not in series:
            series[ind] = []
        series[ind].append((y_pred, y))
    return series # ((tau, num_var) (tau, num_var))


# Community Series generation (context variables are updated at each step with other individuals' predictions)
def generate_series_community(model, dataset, neighbor_graphs, num_var, masked_idxs, close_neighbor_idxs, distant_neighbor_idxs, skip_faults=True):
    series = {i: None for i in set(dataset.individual)}
    device = model.device
    lookback = len(dataset[0][0])
    individual_indexes = {i: 0 for i in set(dataset.individual)}
    individual_series_lengths = dict(Counter(dataset.individual))

    prev_ind = -1
    curr_ind = dataset[0][2]
    with tqdm.tqdm(total=sum(individual_series_lengths)) as pbar:
        while any([individual_indexes[ind] < individual_series_lengths[ind] for ind in individual_indexes.keys()]):
            while individual_indexes[curr_ind] >= individual_series_lengths[curr_ind]:
                curr_ind = list(individual_series_lengths.keys())[(list(individual_series_lengths.keys()).index(curr_ind) + 1) % len(individual_series_lengths)]
            
            x, y, ind = dataset[dataset.individual.index(curr_ind) + individual_indexes[curr_ind]] # individuals are sorted in the dataset
            assert ind == curr_ind, f"individuals are not sorted in the dataset: {ind} != {curr_ind}"

            x = x.to(device)
            y = y.to(device)

            if prev_ind != ind:
                prev_ind = ind

            if series[curr_ind] is not None:
                # Retrieve history from previous prediction instead of the ground truth
                hist = series[ind][-1][0]
                hist = hist.to(device)
                x_obs = x[:,torch.where(torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
                x = torch.cat((hist, x_obs), dim=1)

                # Get neighbors updates if inputs requires predictions from neighbors
                neighbors_window = [] if curr_ind not in neighbor_graphs else neighbor_graphs[curr_ind][individual_indexes[curr_ind]:individual_indexes[curr_ind]+lookback]
                neighbors_updated = True
                x[:,torch.where(torch.tensor([(i in close_neighbor_idxs or i in distant_neighbor_idxs) for i in range(num_var)]))[0]] = 0 # Reset neighbor variables

                for i, (time, close_neighbors, distant_neighbors) in enumerate(neighbors_window):
                    for j, n in enumerate(close_neighbors + distant_neighbors):
                        if n not in series: # If the neighbor is not in the dataset, skip. It can happen if the indivudal has less than lookback sequences
                                if skip_faults:
                                    print(f"Neighbor {n} not in dataset. Skipping...")
                                    continue
                                else:
                                    raise ValueError(f"Neighbor {n} not in dataset.")

                        if series[n] is not None:
                            n_corresponding_index = [k for k, (t, _, _) in enumerate(neighbor_graphs[n]) if t <= time]

                            if len(n_corresponding_index) < 1: # The neighbor sould contain at least one neighbor (the current individual) 
                                if skip_faults:
                                    print(f"cn_corresponding_index should have at least one element, but has {len(n_corresponding_index)} elements. Skipping...")
                                    continue
                                else:
                                    raise ValueError(f"cn_corresponding_index should have at least one element, but has {len(n_corresponding_index)} elements.")

                            n_corresponding_index = n_corresponding_index[-1] - lookback # Select the last index that is before the current time to get the current behaviour of the neighbor

                            if n_corresponding_index < 0: # If the index corresponds to initial data and not generated series values, skip
                                continue

                            if n_corresponding_index < len(series[n]):
                                idxs = close_neighbor_idxs if j < len(close_neighbors) else distant_neighbor_idxs
                                n_obs = series[n][n_corresponding_index][0][-1]
                                n_obs = n_obs.to(device)
                                x[i,torch.where(torch.tensor([k in idxs for k in range(num_var)]))[0]] += n_obs
                            else:
                                neighbors_updated = False
                                curr_ind = n
                                break
                        else:
                            neighbors_updated = False
                            curr_ind = n
                            break
                    if not neighbors_updated:
                        break

                if not neighbors_updated:
                    continue


            # Make prediction
            y_pred = model(x.unsqueeze(0))[0]

            # Remove masked variables
            y_pred = y_pred[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
            y = y[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]
            x = x[:,torch.where(~torch.tensor([i in masked_idxs for i in range(num_var)]))[0]]

            y_pred = F.gumbel_softmax(y_pred.clamp(min=1e-8, max=1-1e-8).log(), hard=True)
            y_pred = torch.cat((x[1:,:], y_pred[-1:,:]), dim=0)

            y_pred = y_pred.cpu()
            y = y.cpu()

            # Update series
            if series[curr_ind] is None:
                series[curr_ind] = []
            series[curr_ind].append((y_pred, y))

            # Update individual index
            individual_indexes[curr_ind] += 1
            pbar.update(1)
        
    return series # ((tau, num_var) (tau, num_var))


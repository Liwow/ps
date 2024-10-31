import os
import warnings
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch_geometric
import random
import time
import warnings
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# this file is to train a predict model. given a instance's bipartite graph as input, the model predict the binary distribution.

# 4 public datasets, IS, WA, CA, IP
# train model task
TaskName = "CA_m_ccon2"
warnings.filterwarnings("ignore")
# set folder
train_task = f'{TaskName}_train'
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain')
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')

# set params
LEARNING_RATE = 0.001
NB_EPOCHS = 9999
BATCH_SIZE = 4
NUM_WORKERS = 0
WEIGHT_NORM = 100

# dataset task
TaskName = "CA_m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR_BG = f'./dataset/{TaskName}/BG'
DIR_SOL = f'./dataset/{TaskName}/solution'
sample_names = os.listdir(DIR_BG)
sample_files = [(os.path.join(DIR_BG, name), os.path.join(DIR_SOL, name).replace('bg', 'sol')) for name in sample_names]
train_files, valid_files = utils.split_sample_by_blocks(sample_files, 0.9, block_size=100)
multimodal = False
if TaskName == "IP":
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy
    from GCN import GraphDataset_position as GraphDataset
elif multimodal:
    from GCN import GraphDataset
    from GCN import GNNPolicy_multimodal as GNNPolicy
else:
    from GCN import GraphDataset_loss as GraphDataset
    from GCN import GNNPolicy_loss as GNNPolicy

train_data = GraphDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=NUM_WORKERS)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUM_WORKERS)

PredictModel = GNNPolicy(TaskName).to(DEVICE)


def lr_lambda(epoch):
    return 0.95 ** ((epoch + 1) // 20)


def EnergyWeightNorm(task):
    if task == "IP":
        return 1
    elif task == "WA":
        return 100
    elif task == "CA" or task == "CA_m" or task == "CA_multi":
        return -4000
    elif task == "beasley":
        return 100
    elif task == "binkar":
        return 1000
    elif task == "ns":
        return 10000
    elif task == "neos":
        return 100
    elif task == "mas":
        return 10000
    elif task == "markshare":
        return 10


def train(predict, data_loader, epoch, optimizer=None, weight_norm=1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    if optimizer:
        predict.train()
        optimizer.zero_grad()
        desc = "Train "
    else:
        predict.eval()
        desc = "Valid "
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(tqdm(data_loader, desc=f"{desc}Epoch {epoch}")):
            batch = batch.to(DEVICE)
            accumulation_steps = 16
            # get target solutions in list format
            solInd = batch.nsols
            con_num = batch.ncons
            label_num = batch.nlabels

            target_sols = []
            target_vals = []
            target_masks = []
            target_labels = []
            target_tights = []
            solEndInd = 0
            valEndInd = 0
            conEndInd = 0
            labelEndInd = 0

            for i in range(solInd.shape[0]):  # for in batch
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                conStartInd = conEndInd
                conEndInd = solInd[i] * con_num[i] + conStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                labelStartInd = labelEndInd
                labelEndInd = solInd[i] * label_num[i] + labelStartInd
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                cons = batch.c_labels[labelStartInd:labelEndInd].reshape(-1, label_num[i])
                masks = batch.c_mask[conStartInd:conEndInd].reshape(-1, con_num[i])[0]
                vals = batch.objVals[valStartInd:valEndInd]

                tights = batch.t_labels[labelStartInd:labelEndInd].reshape(-1, label_num[i])
                target_tights.append(tights)

                target_sols.append(sols)
                target_labels.append(cons)
                target_masks.append(masks)
                target_vals.append(vals)

            mask = torch.cat(target_masks)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # remove nan value
            # predict the binary distribution, BD
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                mask
            )
            predict_con, predict_sol, predict_t = BD

            # compute loss
            loss = 0
            index_arrow = 0
            index_cons = 0
            # in batch
            for ind, (sols, vals, labels, tights) in enumerate(
                    zip(target_sols, target_vals, target_labels, target_tights)):
                # compute weight
                n_vals = vals
                exp_weight = torch.exp(-n_vals / weight_norm)
                weight = exp_weight / exp_weight.sum()

                # get a binary mask
                varInds = batch.varInds[ind]
                varname_map = varInds[0][0]
                b_vars = varInds[1][0].long()

                # get binary variables
                sols = sols[:, varname_map][:, b_vars]
                pre_cons = predict_con[index_cons: index_cons + label_num[ind]].squeeze()
                pre_t = predict_t[index_cons: index_cons + label_num[ind]].squeeze()
                index_cons = index_cons + label_num[ind]
                # todo con_loss
                pos_loss = -(pre_cons + 1e-8).log()[None, :] * (labels == 1).float()
                neg_loss = -(1 - pre_cons + 1e-8).log()[None, :] * (labels == 0).float()
                masked_con_loss = (pos_loss + neg_loss) * weight[:, None]

                pos_loss = -(pre_t + 1e-8).log()[None, :] * (tights == 1).float()
                neg_loss = -(1 - pre_t + 1e-8).log()[None, :] * (tights == 0).float()
                tight_loss = (pos_loss + neg_loss) * weight[:, None]
                loss += (2 * masked_con_loss.sum())
                loss += (4 * tight_loss.sum())

                # todo penalty
                # masked_con_loss = utils.focal_loss(pre_cons, labels, weight)
                # penalty = 300 * torch.where(labels == 0, -torch.log(1 - pre_cons + 1e-8), torch.zeros_like(pre_cons))
                # if epoch > 200:
                #     loss += (20 * masked_con_loss + penalty.mean())
                # else:
                #     loss += (20 * masked_con_loss)

                n_var = batch.ntvars[ind]
                pre_sols = predict_sol[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var
                pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols == 1).float()
                neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols == 0).float()
                sum_loss = pos_loss + neg_loss
                sample_sols_loss = sum_loss * weight[:, None]
                loss += sample_sols_loss.sum()
            if optimizer is not None:
                loss.backward()
            if optimizer is not None and (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if optimizer is not None and step == len(data_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss


optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)
scheduler = LambdaLR(optimizer, lr_lambda)
weight_norm = EnergyWeightNorm(TaskName) if not None else 100
best_val_loss = 99999
for epoch in range(NB_EPOCHS):

    begin = time.time()
    train_loss = train(PredictModel, train_loader, epoch, optimizer, weight_norm)
    scheduler.step()
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
    valid_loss = train(PredictModel, valid_loader, epoch, None, weight_norm)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(PredictModel.state_dict(), model_save_path + 'model_best.pth')
    torch.save(PredictModel.state_dict(), model_save_path + 'model_last.pth')
    st = f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{valid_loss}    TIME:{time.time() - begin}\n'
    log_file.write(st.encode())
    log_file.flush()
print('done')
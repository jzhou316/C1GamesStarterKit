import os
import pickle
import random
import torch
import torch.nn as nn
from copy import deepcopy
from model import QModel

# ============================ parameters =======================
s_dim = 3136  # 28 * 28 * 4
a_att_dim = 840  # 28 * 3 * 10
a_def_dim = 1568  # 28 * 14 * 4
# total 4368, #params 4368 * 500 = 2,184,000
hid_dim = 500

data_dir = '../../data'
num_runs = 100

lr = 3e-4
clip = 5
bsz = 100
epochs = 100
save_dir = 'saved_models'
save_name = 'model.pt'
old_model_path = 'saved_models_old/model'
# ===============================================================


def train(old_model, model, optimizer, data_list, idx, bsz=1):
    """train for 1 epoch"""
    running_loss = 0
    running_reward = 0

    num_batches = len(idx) // bsz
    if len(idx) % bsz > 0:
        num_batches += 1
    batch_idx = [idx[i * bsz : (i + 1) * bsz] for i in range(num_batches)]

    for bdx in batch_idx:
        optimizer.zero_grad()
        # bdx is a list
        data_cur_batch = [data_list[i] for i in bdx]
        data_nxt_batch = [data_list[i + 1] for i in bdx]
        s_cur_batch = torch.stack([d[0].view(-1) for d in data_cur_batch], dim=0).float().cuda()
        att_cur_batch = torch.stack([d[1].view(-1) for d in data_cur_batch], dim=0).float().cuda()
        def_cur_batch = torch.stack([d[2].view(-1) for d in data_cur_batch], dim=0).float().cuda()
        s_nxt_batch = torch.stack([d[0].view(-1) for d in data_nxt_batch], dim=0).float().cuda()
        att_nxt_batch = torch.stack([d[1].view(-1) for d in data_nxt_batch], dim=0).float().cuda()
        def_nxt_batch = torch.stack([d[2].view(-1) for d in data_nxt_batch], dim=0).float().cuda()
        reward_batch = torch.tensor([(dn[3] - dn[4]) - (dc[3] - dc[4]) for dn, dc in zip(data_nxt_batch, data_cur_batch)]).view(-1, 1).float().cuda()

        loss = (model.q_val(s_cur_batch, att_cur_batch, def_cur_batch) - old_model.q_val(s_nxt_batch, att_nxt_batch, def_nxt_batch).detach() - reward_batch) ** 2
        loss = loss.sum()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), clip) 
        optimizer.step()
        running_reward += (reward_batch ** 2).sum().item()
        running_loss += loss.item()

    print('running reward: ', running_reward / len(idx))

    return running_loss / len(idx)

    '''
    for i in idx:
        optimizer.zero_grad()
        data_cur = data_list[i]
        data_nxt = data_list[i + 1]
        reward = (data_nxt[3] - data_nxt[4]) - (data_cur[3] - data_cur[4])
        loss = (model.q_val(data_cur[0].float().cuda(), data_cur[1].float().cuda(), data_cur[2].float().cuda()) - \
                old_model.q_val(data_nxt[0].float().cuda(), data_nxt[1].float().cuda(), data_nxt[2].float().cuda()).detach() - reward) ** 2
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), clip) 
        optimizer.step()
        running_reward += reward**2
        running_loss += loss.item()
    print (running_reward/len(idx))

    return running_loss / len(idx)
    '''

if __name__ == '__main__':
    data_files = ['run_' + str(i) + '.pkl' for i in range(num_runs)]
    # read in all the data
    data_list = []
    for n in range(num_runs):
        if os.path.exists(os.path.join(data_dir, data_files[n])):
            data_list += pickle.load(open(os.path.join(data_dir, data_files[n]), 'rb'))

    # initialize the model
    old_model = QModel(s_dim, a_att_dim, a_def_dim, hid_dim)
    if len(old_model_path) > 0:
        print ('loading')
#        old_model.load_state_dict(torch.load(old_model_path))
    #    sys.exit(1)
    old_model.eval()
    model = deepcopy(old_model)
    model.train()
    model.cuda()
    old_model.cuda()

    # optimizers
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    num_train = len(data_list) - 1
    train_idx = list(range(num_train))
    for ep in range(epochs):
        print(f'epoch {ep + 1}/{epochs} ' + '-' * 10)
        random.shuffle(train_idx)
        #TODO: THIS has the HIGH VAIRANCE PROBLEM!
        loss = train(old_model, model, optimizer, data_list, train_idx, bsz=bsz)
        print(f'loss: {loss:.5f}')

    # save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))

import os
import pickle
import torch
import torch.nn as nn

from model import QModel

# ============================ parameters =======================
s_dim = 2352  # 28 * 28 * 3
a_att_dim = 840  # 28 * 3 * 10
a_def_dim = 1176  # 28 * 14 * 3
# total 4368, #params 4368 * 500 = 2,184,000
hid_dim = 500

lr = 1
save_dir = '../saved_models'
save_name = 'model'


# ===============================================================


def train_1file(model, optimizer, data_list):
    """train for 1 .pkl file"""
    num_data = len(data_list)
    for i in range(num_data - 1):
        optimizer.zero_grad()
        data_cur = data_list[i]
        data_nxt = data_list[i + 1]
        reward = (data_nxt[3] - data_nxt[4]) - (data_cur[3] - data_nxt[4])
        loss = model.q_val(data_cur[0], [data_cur[1], data_cur[2]]) - \
               model.q_val(data_nxt[0], [data_nxt[1], data_nxt[2]]).detach() + reward
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':
    data_dir = './'
    num_runs = 100
    data_files = ['run_' + str(i) + '.pkl' for i in range(num_runs)]

    # initialize the model
    model = QModel(s_dim, a_att_dim, a_def_dim, hid_dim)

    # optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # training loop
    for n in range(num_runs):
        data_list = pickle.load(open(os.path.join(data_dir, data_files[n]), 'rb'))
        loss = train_1file(model, optimizer, data_list)

    # save model
    torch.save(model, os.path.join(save_dir, save_name))

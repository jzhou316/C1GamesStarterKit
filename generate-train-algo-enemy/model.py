import torch
import torch.nn as nn
import sys

class QModel(nn.Module):
    """model for Q-learning for game action prediction."""
    def __init__(self, s_dim, a_att_dim, a_def_dim, hid_dim):
        super().__init__()
        self.s_dim = s_dim
        self.a_att_dim = a_att_dim
        self.a_def_dim = a_def_dim
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(nn.Linear(s_dim + a_att_dim + a_def_dim, hid_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(hid_dim, 1))
        self.reset_parameters()
        print(f'number of parameters: {self.param_size()}', file=sys.stderr)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.05, 0.05)

    def q_val(self, s, a_att, a_def):
        """
        :param self:
        :param s: current state of the map, represented by a tensor
        :param a: action, represented by a tensor
        :return:
        the Q value
        """
        if s.dim() > 1:
            s = s.reshape(-1)
        if a_att.dim() > 1:
            a_att = a_att.view(-1)
        if a_def.dim() > 1:
            a_def = a_def.view(-1)
        q = self.mlp(torch.cat([s, a_att, a_def], dim=0))
        return q

    def infer_act(self, s): # recursive?
        """
        :param self:
        :param s: current state of the map, represented by a tensor
        :return:
        action, represented by a tensor
        """
        if s.dim() > 1:
            s = s.reshape(-1)
        a_att = torch.zeros(self.a_att_dim, requires_grad=True)
        a_def = torch.zeros(self.a_def_dim, requires_grad=True)
        a = [a_att, a_def]
        q = self.q_val(s, a_att, a_def)
        a_grads = torch.autograd.grad(q, [a_att, a_def])
        a_att.requires_grad = False
        a_def.requires_grad = False
        for aa, grad in zip(a, a_grads):
            grad_sorted, grad_idx = torch.sort(grad)
            idx = grad_idx[grad_sorted > 0]
            aa[idx] = 1

        return a_att.reshape(28, 3, 10), a_def.reshape(28, 14, 4), a_grads[0].reshape(28, 3, 10), a_grads[1].reshape(28, 14, 4)

    def param_size(self):
        return sum([p.numel() for p in self.parameters()])

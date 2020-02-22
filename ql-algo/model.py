import torch
import torch.nn as nn


class QModel(nn.Module):
    """model for Q-learning for game action prediction."""
    def __init__(self, s_dim, a_att_dim, a_def_dim, hid_dim):
        super().__init__()
        self.s_dim = s_dim
        self.a_att_dim = a_att_dim
        self.a_def_dim = a_def_dim
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(nn.Linear(s_dim + a_att_dim + a_def_dim, hid_dim),
                                 nn.Relu(),
                                 nn.Linear(hid_dim, 1))

    def q_val(self, s, a):
        """
        :param self:
        :param s: current state of the map, represented by a tensor
        :param a: action, represented by a tensor
        :return:
        the Q value
        """
        q = self.mlp(torch.cat([s, *a], dim=0))
        return q

    def infer_act(self, s):
        """
        :param self:
        :param s: current state of the map, represented by a tensor
        :return:
        action, represented by a tensor
        """
        a_att = torch.zeros(self.a_att_dim)
        a_def = torch.zeros(self.a_def_dim)
        a = [a_att, a_def]
        q = self.q_val(s, a)
        a_grads = torch.autograd.grad(q, [a_att, a_def])[0]
        for aa, grad in zip(a, a_grads):
            grad_sorted, grad_idx = torch.sort(grad)
            idx = grad_idx[grad_sorted > 0]
            aa[idx] = 1

        return a_att.reshape(28, 3, 10), a_def.reshape(28, 14, 3)

    def param_size(self):
        return sum([p.numel() for p in self.parameters()])

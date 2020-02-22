import torch
import torch.nn as nn


class QModel(nn.Module):
    """model for Q-learning for game action prediction."""
    def __init__(self, s_dim, a_dim, hid_dim):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(nn.Linear(s_dim + a_dim, hid_dim),
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
        q = self.mlp(torch.cat([s, a], dim=0))
        return q

    def infer_act(self, s):
        """
        :param self:
        :param s: current state of the map, represented by a tensor
        :return:
        action, represented by a tensor
        """
        a = torch.zeros(self.a_dim)
        q = self.q_val(s, a)
        grad = torch.autograd.grad(q, a)[0]
        grad_sorted, grad_idx = torch.sort(grad)
        idx = grad_idx[grad_sorted > 0]
        a[idx] = 1

        return a

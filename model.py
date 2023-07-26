from hyp import Hyp
import numpy as np
from opt_hyp import opt_hyp


class Model:

    def __init__(self, hyp=Hyp(),
                 lb=np.concatenate((1e-6 * np.ones((4, 1)), -10 * np.ones((1, 1)))),
                 ub=np.concatenate((1000 * np.ones((4, 1)), np.zeros((1, 1)))),
                 X0=None, dX0=None, JRX0=None):
        self.hyp_ = hyp
        self.lb_ = lb
        self.ub_ = ub
        self.X_ = X0
        self.dX_ = dX0
        self.JRX_ = JRX0

    def get_Hyp(self):
        return self.hyp_

    def get_Hyp_sd(self):
        return self.hyp_.get_SD()

    def get_Hyp_sn(self):
        return self.hyp_.get_SN()

    def get_Hyp_l(self):
        return self.hyp_.get_L()

    def get_Hyp_JRvec(self):
        return self.hyp_.get_JRvec()

    def optimize_Hyp(self):
        print('optimizing hyperparameters...')
        self.hyp_ = opt_hyp(self.hyp_, self.lb_, self.ub_, self.X_, self.dX_)

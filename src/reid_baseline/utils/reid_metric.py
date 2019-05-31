# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import pickle
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func

class Distmat(Metric):
    def __init__(self, num_query, max_rank=50):
        super(Distmat, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.paths = []

    def update(self, output):
        feat, pid, camid, path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths.extend(path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_paths = self.paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_paths = self.paths[self.num_query:]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        print(distmat.shape)
        print(len(q_paths),len(g_paths))
        
        f_mat = open('result/distmat.mat','wb')
        f_mat.write(pickle.dumps(distmat)) 
        f_mat.close()
        f_q = open('result/q_path.pickle','wb') 
        f_q.write(pickle.dumps(q_paths))
        f_q.close()

        f_g = open('result/g_path.pickle','wb')
        f_g.write(pickle.dumps(g_paths))
        f_g.close()

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

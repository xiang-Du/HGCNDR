from torch import nn
from torch.nn import init
import torch
from torch.nn import functional as fn
from layer import *
import numpy as np
from utils import calc_norm


class HGCNDR(nn.Module):
    def __init__(self, drug_similarity, disease_similarity, output_dim1, output_dim2, incidence, layer_num, k=20, variable_weight=True, drop=0.2, order=2, score='bi', mode='separated', device='cpu') -> None:
        super(HGCNDR, self).__init__()
        assert output_dim2 % (order) == 0, 'output_dim is not suitable'
        drug_num = drug_similarity.shape[0]
        self.drug_num = drug_num
        disease_num = disease_similarity.shape[0]
        self.disease_num = disease_num
        self.drug_similarity = torch.tensor(
            drug_similarity, dtype=torch.float32, device=device)
        self.disease_similarity = torch.tensor(
            disease_similarity, dtype=torch.float32, device=device)
        self.incidence = torch.tensor(
            incidence, dtype=torch.float32, device=device)
        self.layer_num = layer_num
        self.device = device
        drug_attr = torch.tensor(np.concatenate((drug_similarity, np.zeros(
            (drug_similarity.shape[0], disease_similarity.shape[1]))), axis=1), dtype=torch.float32, device=device)
        disease_attr = torch.tensor(np.concatenate((np.zeros(
            (disease_similarity.shape[0], drug_similarity.shape[1])), disease_similarity), axis=1), dtype=torch.float32, device=device)
        self.init_feature = torch.concat((drug_attr, disease_attr), dim=0)
        if score == 'mlp':
            self.decoder = MLPDecoder(output_dim1 + output_dim2, drug_num)
        elif score == 'bi':
            self.decoder = BilinearDecoder(output_dim1 + output_dim2, drug_num)
        self.k = k
        self.dr_dr_gcn = GCN(
            drug_num, drug_attr.shape[1], output_dim1, layer_num, drop, layer_att=False, norm='b')
        self.di_di_gcn = GCN(
            disease_num, disease_attr.shape[1], output_dim1, layer_num, drop, layer_att=False, norm='b')
        self._preprocess()
        self.hypergraph = HyperGraph(drug_num, disease_num, drug_attr.shape[1], output_dim2 // (
            order), self.incidence, layer_num, variable_weight=variable_weight, layer_att=False, order=order, res=True, drop=0.2, mode=mode, device=device)
        # self.hypergraph = HyperGraph(drug_num, disease_num, drug_attr.shape[1], output_dim // (
        #     order), self.incidence, layer_num, variable_weight=variable_weight, layer_att=False, order=order, res=True, drop=0.2, device=device)

    def _select_topk(self, mode='dr', k=-1):
        if mode == 'dr':
            data = self.drug_similarity
        else:
            data = self.disease_similarity
        if k <= 0:
            return data
        assert k <= data.shape[1]
        _, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * \
            torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1).to(device=data.device)
        new_data = torch.zeros_like(data)
        new_data[row, col] = data[row, col]
        # new_data[row, col] = 1.0
        return new_data

    def _preprocess(self):
        # construct similarity network
        dr_dr_s = self._select_topk('dr', self.k)
        temp_dr = torch.sqrt(dr_dr_s * dr_dr_s.T)
        temp_dr = dr_dr_s - temp_dr
        dr_dr_s = dr_dr_s + temp_dr.T
        # temp_dr = dr_dr_s.T.clone()
        # temp_dr[dr_dr_s == temp_dr] = 0
        # dr_dr_s = dr_dr_s + temp_dr
        dr_dv = torch.sum(dr_dr_s, dim=1, keepdim=True)
        invdr_dv = calc_norm(dr_dv, -0.5)
        self.dr_dr_s =  invdr_dv * dr_dr_s * invdr_dv.T

        di_di_s = self._select_topk('di', self.k)
        temp_di = torch.sqrt(di_di_s * di_di_s.T)
        temp_di = di_di_s - temp_di
        di_di_s = di_di_s + temp_di.T
        # temp_di = di_di_s.T.clone()
        # temp_di[di_di_s == temp_di] = 0
        # di_di_s = di_di_s + temp_di
        di_dv = torch.sum(di_di_s, dim=1, keepdim=True)
        invdi_dv = calc_norm(di_dv, -0.5)
        self.di_di_s =  invdi_dv * di_di_s * invdi_dv.T

    def get_embedding(self):
        drug_s = self.dr_dr_gcn(
            self.dr_dr_s, self.init_feature[:self.drug_num], activation=torch.relu)
        disease_s = self.di_di_gcn(
            self.di_di_s, self.init_feature[self.drug_num:], activation=torch.relu)
        feature_s = torch.concat((drug_s, disease_s), dim=0)
        feature_h = self.hypergraph(self.init_feature)
        embedding = torch.concat((feature_s, feature_h), dim=1)
        # embedding  = feature_h
        return embedding, feature_s, feature_h

    def _predict(self, h, t, n_s):

        embedding, feature_s, feature_h = self.get_embedding()
        pos_score, neg_score = self.decoder(embedding, h, t, n_s)
        return pos_score, neg_score, feature_s, feature_h

    def predict(self, h, t):
        embedding, _, _ = self.get_embedding()
        score = self.decoder.predict(embedding, h, t)
        
        return score

    def forward(self, h, t, n_s):
        return self._predict(h, t, n_s)

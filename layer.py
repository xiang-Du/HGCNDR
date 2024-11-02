from torch import nn
import torch
import torch.nn.init as init
from utils import calc_norm, splits
import itertools
import numpy as np


class Normal(nn.Module):
    def __init__(self, feature_shape, eps=1e-10,mode='b'):
        super(Normal, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_shape))
        self.bias = nn.Parameter(torch.zeros(feature_shape))
        self.eps = eps
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l':
            dim = -1
        else:
            dim = 0
        mean = x.mean(dim, keepdim=True)
        std = x.std(dim, keepdim=True)
        return self.gamma * (x - mean) / torch.pow((std + self.eps), 1) + self.bias


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, drop, use_bias):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # self.res = res
        # if res:
        #     if input_dim != output_dim:
        #         self.res_weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        # self.normal = Normal((num, output_dim))
        self.drop = nn.Dropout(drop)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_normal_(self.weight)
        # if self.res:
        #     if self.input_dim != self.output_dim:
        #         nn.init.xavier_normal_(self.res_weight, gain=nn.init.calculate_gain('relu'))

    def _conv(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def forward(self, g, x, activation):
        # o_x = x
        x = self.drop(x)
        result = self._conv(g, x)
        result = activation(result)
        return result


class GCN(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, layer_num, drop, layer_att=True, res=True, use_bias=True, norm='l') -> None:
        super(GCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.layer_num = layer_num
        self.layer_att = layer_att
        self.res = res
        self.input_dim, self.output_dim = input_dim, output_dim
        if res:
            if input_dim != output_dim:
                self.res_weight = nn.Parameter(
                    torch.Tensor(input_dim, output_dim))
        if layer_att:
            self.alpha = nn.Parameter(torch.tensor(
                [1/layer_num]*layer_num).reshape(layer_num, 1, 1))
        self.gcn_layers.append(GraphConvolution(
            input_dim, output_dim, drop, use_bias))
        for _ in range(layer_num-1):
            self.gcn_layers.append(GraphConvolution(
                output_dim, output_dim, drop, use_bias))
        self.drop = nn.Dropout(drop)
        self.normal = Normal((node_num, output_dim), mode=norm)

        self._reset_parameter()

    def _reset_parameter(self):
        if self.res:
            if self.input_dim != self.output_dim:
                nn.init.xavier_normal_(self.res_weight)

    def forward(self, g, x, activation=torch.tanh):
        o_x = x
        features = []
        for gcn in self.gcn_layers:
            x = gcn(g, x, activation)
            features.append(x)
        
        if self.layer_att:
            result = torch.stack(features, dim=0)
            result = self.alpha * result
            result = torch.sum(result, dim=0)
        else:
            result = features[-1]
        if self.res:
            if self.input_dim != self.output_dim:
                o_x = torch.matmul(o_x, self.res_weight)
            result = o_x + result
        result = self.drop(self.normal(result))
        return result


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class HyperGraphconv(nn.Module):
    def __init__(self, drug_num, disease_num, input_dim, output_dim, incidence, variable_weight, layer_att, order, drop, mode, device):
        super(HyperGraphconv, self).__init__()
        self.drug_num = drug_num
        self.disease_num = disease_num
        self.incidence = incidence
        self.variable_weight = variable_weight
        self.order = order
        self.layer_att = layer_att
        self.device = device
        self.intra_adj = None
        self.mode = mode
        
        self.drop = nn.Dropout(drop)
        if variable_weight:
            self.w_matrix = nn.Parameter(torch.ones((1, incidence.shape[1])))
            # self.w_matrix = nn.Parameter(torch.tensor([0.5] * incidence.shape[1]).reshape(1, incidence.shape[1]))
        else:
            self.w_matrix = torch.ones(
                (1, incidence.shape[1]), dtype=torch.float32, device=device)
        if layer_att:
            self.alpha = nn.Parameter(torch.tensor(
                [1/(order+1)]*(order+1)).reshape((order+1), 1, 1))
        if mode == 'separated':
            self.dr_gcn = GCN(drug_num, input_dim, output_dim, 1, drop, layer_att=False, res=False, norm='b')
            self.di_gcn = GCN(disease_num, input_dim, output_dim, 1, drop, layer_att=False, res=False, norm='b')
        
        self.drdi_gcn = GCN(drug_num + disease_num,
                            input_dim, output_dim, 1, drop, layer_att=False, res=False, norm='b')
        self.intra_w = nn.ParameterList(
            [nn.Parameter(torch.Tensor(input_dim, output_dim)) for _ in range(order-1)])
        self.inter_r = nn.ParameterList(
            [nn.Parameter(torch.Tensor(output_dim, output_dim)) for _ in range(order-1)])
        # self.attention = nn.ModuleList(Attention(output_dim) for _ in range(order-1))
        self._reset_parameter()
        self._preprocess()

    def _reset_parameter(self):
        for intra_w in self.intra_w:
            nn.init.xavier_normal_(intra_w)
        for inter_r in self.inter_r:
            nn.init.xavier_normal_(inter_r)

    def _preprocess(self):
        self.adj = self.incidence + \
            torch.eye(self.incidence.shape[0], device=self.device)
        # calculate degree of hyperedge shape(1,n)
        de = torch.sum(self.adj, dim=0, keepdim=True)
        self.invde = calc_norm(de, -1)
        

    def _generate_adj_from_hypergraph(self):
        if self.intra_adj is not None:
            return self.intra_adj
        h = self.adj
        h_t = self.adj.T
        h = h * self.w_matrix
        dv = torch.sum(h, dim=1, keepdim=True)
        invdv = calc_norm(dv, -0.5)
        h = h * self.invde
        adj = torch.matmul(h, h_t)
        adj = invdv * adj * invdv.T
        if not self.variable_weight:
            self.intra_adj = adj
        return adj

    def _decompose_adj(self):
        
        adj = self._generate_adj_from_hypergraph()
        dr_dr = torch.zeros(self.drug_num, self.drug_num, device=self.device)
        di_di = torch.zeros(self.disease_num, self.disease_num, device=self.device)
        dr_di = torch.zeros(self.drug_num + self.disease_num, self.drug_num + self.disease_num, device=self.device)
        dr_dr = dr_dr + adj[:self.drug_num, :self.drug_num]
        di_di = di_di + adj[self.drug_num:, self.drug_num:]
        dr_di[:self.drug_num, self.drug_num:] = dr_di[:self.drug_num, self.drug_num:] + adj[:self.drug_num, self.drug_num:]
        dr_di[self.drug_num:, :self.drug_num] = dr_di[self.drug_num:, :self.drug_num] + adj[self.drug_num:, :self.drug_num]

        return dr_dr, di_di, dr_di

    def n_order_featrue(self, x, n):
        assert n >= 2 and n <= 4, "order n is invalid"
        x = self.drop(x)
        feature_intra = x @ self.intra_w[n-2]
        
        drug = feature_intra[:self.drug_num]
        disease = feature_intra[self.drug_num:]
        
     

        alpha = self.adj
        norm = torch.sum(self.adj, dim=1, keepdim=True)
        x = norm
        for i in range(n, 1, -1):
            x = x - 1
            norm = norm * x / i
        norm = calc_norm(norm, -1)

        
        alpha = self.incidence
        first_order = alpha @ feature_intra
        if n == 2:
            order1_dr = first_order[:self.drug_num]
            order1_di = first_order[self.drug_num:]
            # inter domain
            order1_dr = (drug @ self.inter_r[n-2]) * order1_dr
            order1_di = (order1_di @ self.inter_r[n-2]) * disease
            order1 = torch.concat((order1_dr, order1_di), dim=0)
            # intra domain
            order2 = (first_order ** 2 - alpha @ (feature_intra ** 2)) / 2
        elif n == 3:
            
            second_order = (first_order ** 2 - alpha @ (feature_intra ** 2)) / 2
            
            # inter domain
            order1_dr = second_order[:self.drug_num]
            order1_di = second_order[self.drug_num:]
            order2_dr = (drug @ self.inter_r[n-2]) * order1_dr
            order2_di = (order1_di @ self.inter_r[n-2]) * disease
            order1 = torch.concat((order2_dr, order2_di), dim=0)
            # intra domain
            order2 = ((first_order ** 3) - 3*(second_order *
                            first_order) - alpha@((feature_intra)**3)) / (-3)
        else:
            second_order = (first_order ** 2 - alpha @ (feature_intra ** 2)) / 2
            third_order = ((first_order ** 3) - 3*(second_order * first_order) -
                           alpha @ ((feature_intra)**3)) / (-3)
            # inter domain
            order1_dr = third_order[:self.drug_num]
            order1_di = third_order[self.drug_num:]
            order2_dr = (drug @ self.inter_r[n-2]) * order1_dr
            order2_di = (order1_di @ self.inter_r[n-2]) * disease
            order1 = torch.concat((order2_dr, order2_di), dim=0)
            # intra domain
            a2b2 = ((alpha@(feature_intra ** 2)) **
                    2 - alpha@(feature_intra ** 4)) / 2

            order2 = ((second_order ** 2) -
                            (2 * third_order * first_order) - a2b2) / (-2)
            
        # mask = torch.all(order == 0, dim=1) == False
        # norm_x = torch.zeros_like(order)
        # norm_x[mask] = order[mask] / torch.norm(order[mask], dim=1, p=2, keepdim=True)
        # order = norm_x
        # order = order / torch.norm(order, dim=1, p=2, keepdim=True)
        order = norm * (order1 + order2)
        # order_temp = torch.stack((order1, order2), dim=1)
        # order, _ = self.attention[n-2](order_temp)
        return order

    def forward(self, x):
        
        if self.mode == 'separated':
            drug = x[:self.drug_num]
            disease = x[self.drug_num:]
            dr_dr, di_di, dr_di = self._decompose_adj()
            drug_intra = self.dr_gcn(dr_dr, drug, activation=lambda x: x)
            disease_intra = self.di_gcn(di_di, disease, activation=lambda x: x)
            f_intra = torch.concat((drug_intra, disease_intra), dim=0)
            f_inter = self.drdi_gcn(dr_di, x, activation=lambda x: x)
            temp = [torch.tanh(f_intra + f_inter)]
        else:
            dr_di = self._generate_adj_from_hypergraph()
            f_inter = self.drdi_gcn(dr_di, x, activation=torch.tanh)
            temp = [f_inter]
        order = [self.n_order_featrue(x, n)
                 for n in range(2, self.order+1)]
        temp = temp + order
        # embedding = torch.stack(temp, dim=0)
        # if self.layer_att:
        #     embedding = self.alpha * embedding
        #     embedding = torch.sum(embedding, dim=0)
        # else:
        #     embedding = torch.mean(embedding, dim=0)
        embedding = torch.concat(temp, dim=1)
        
        # embedding = self.normal(embedding)
        return embedding


class HyperGraph(nn.Module):
    def __init__(self, drug_num, disease_num, input_dim, output_dim, incidence, layer_num=2, variable_weight=True, layer_att=True, order=2, res=True, drop=0.2,mode='separated', device='cpu') -> None:
        super(HyperGraph, self).__init__()
        self.drug_num = drug_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.disease_num = disease_num
        self.res = res
        self.res_weight = None
        self.drop = nn.Dropout(drop)
        self.normal = Normal((drug_num + disease_num, output_dim * order), mode='b')
        self.layer_att = layer_att
        if layer_att:
            self.alpha = nn.Parameter(torch.tensor(
                [1/layer_num]*layer_num).reshape(layer_num, 1, 1))
        if res:
            if input_dim != output_dim * order:
                self.res_weight = nn.Parameter(torch.Tensor(input_dim, output_dim * order))

        self.hypergraphconvs = nn.ModuleList()
        hypergraphconv = HyperGraphconv(drug_num, disease_num, input_dim, output_dim,
                                        incidence, variable_weight, False, order, drop, mode, device)
        self.hypergraphconvs.append(hypergraphconv)
        for _ in range(layer_num-1):
            hypergraphconv = HyperGraphconv(drug_num, disease_num, output_dim * (order), output_dim,
                                        incidence, variable_weight, False, order, drop, mode, device)
            self.hypergraphconvs.append(hypergraphconv)
        self._reset_parameter()
        
    def _reset_parameter(self):
        if self.res_weight is not None:
            nn.init.xavier_normal_(self.res_weight)
    

    def forward(self, x):
        x_o = x
        features = []
        for hypergraphconv in self.hypergraphconvs:
            x = hypergraphconv(x)
            features.append(x)
        
        if self.layer_att:
            result = torch.stack(features, dim=0)
            result = self.alpha * result
            result = torch.sum(result, dim=0)
        else:
            result = features[-1]
        if self.res:
            if self.input_dim != self.output_dim:
                x_o = torch.matmul(x_o, self.res_weight)

                result = x_o + result
        result = self.drop(self.normal(result))
        return result


class MLPDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 drug_num,
                 dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.drug_num = drug_num
        self.score_function = nn.Sequential(nn.Linear(2 * input_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        for i in self.score_function:
            if isinstance(i, nn.Linear) :
                i.reset_parameters()
    
    def predict(self, embed, drug, disease):
        drug_embed = embed[drug]
        disease_embed = embed[disease]
        rep = torch.concat((drug_embed, disease_embed), dim=-1)
        result = self.score_function(rep)
        # result [n]
        result = result.squeeze(-1)
        return result
        
    def denovo(self, embed, drug):
        drug_embed = embed[[drug]]
        disease_embed = embed[self.drug_num:]
        drug_embed_r = drug_embed.repeat(disease_embed.shape[0], 1)
        rep = torch.concat((drug_embed_r, disease_embed), dim=-1)
        result = self.score_function(rep)
        # result [n]
        result = result.squeeze(-1)
        return result

    def forward(self, embed, h, t, n_s):
        if torch.max(n_s<self.drug_num) == True:
            n_t, n_h =  torch.chunk(n_s, 2, dim=1)
        else:
            n_t = n_s
            n_h = None
        h_embed = embed[h]
        t_embed = embed[t]
        h_embed_r1 = h_embed.repeat(1, t.shape[1], 1)
        pos_sample = torch.concat((h_embed_r1, t_embed), dim=-1)
        n_t_embed = embed[n_t]
        neg_num = n_t.shape[1]
        h_embed_r2 = h_embed.repeat(1, neg_num, 1)
        neg_sample1 = torch.concat((h_embed_r2, n_t_embed), dim=-1)
        pos_score = self.score_function(pos_sample)
        neg_score = self.score_function(neg_sample1)
          
        if n_h is not None:
            n_h_embed = embed[n_h]
            t_embed_r = t_embed.repeat(1, neg_num, 1)
            neg_sample2 = torch.concat((n_h_embed, t_embed_r), dim=-1)
            neg_score2 = self.score_function(neg_sample2)
            neg_score = torch.concat((neg_score, neg_score2), dim=1)
        pos_score = pos_score.squeeze(dim=-1)
        neg_score = neg_score.squeeze(dim=-1)
        # pos_score [n,1] neg_score [n, neg_ratio]
        return pos_score, neg_score

class BilinearDecoder(nn.Module):
    def __init__(self, input_dim, drug_num):
        super(BilinearDecoder, self).__init__()
        self.drug_num = drug_num
        self.input_dim = input_dim
        self.drug_num =drug_num
        self.score_matrix = nn.Parameter(torch.Tensor(input_dim, input_dim ))
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_normal_(self.score_matrix)

    def forward(self, embed:torch.Tensor, h, t, n_s):
        if torch.max(n_s < self.drug_num) == True:
            n_t, n_h =  torch.chunk(n_s, 2, dim=1)
        else:
            n_t = n_s
            n_h =   None
        h_embed = embed[h]
        t_embed = embed[t]
        # pos_score [n,1]
        h_embed = torch.matmul(h_embed, self.score_matrix)
        pos_score = torch.multiply(h_embed, t_embed).sum(-1)
        n_t_embed = embed[n_t]
        neg_score = (h_embed * n_t_embed).sum(-1)
        if n_h is not None:
            n_h_embed = embed[n_h]
            n_h_embed = torch.matmul(n_h_embed, self.score_matrix)
            neg_score2 = torch.multiply(n_h_embed, t_embed).sum(-1)
            neg_score = torch.concat((neg_score, neg_score2), dim=1)
        # pos_score [n,1] neg_score [n, neg_ratio]    
        return pos_score, neg_score
    
class DotDecoder(nn.Module):
    def __init__(self, drug_num):
        super(BilinearDecoder, self).__init__()
        self.drug_num = drug_num
    
    def forward(self, embed:torch.Tensor, h, t, n_s):
        if torch.max(n_s < self.drug_num) == True:
            n_t, n_h =  torch.chunk(n_s, 2, dim=1)
        else:
            n_t = n_s
            n_h =   None
        h_embed = embed[h]
        t_embed = embed[t]
        # pos_score [n,1]
        pos_score = torch.multiply(h_embed, t_embed).sum(-1)
        n_t_embed = embed[n_t]
        neg_score = (h_embed * n_t_embed).sum(-1)
        if n_h is not None:
            n_h_embed = embed[n_h]
            neg_score2 = torch.multiply(n_h_embed, t_embed).sum(-1)
            neg_score = torch.concat((neg_score, neg_score2), dim=1)
        # pos_score [n,1] neg_score [n, neg_ratio]    
        return pos_score, neg_score

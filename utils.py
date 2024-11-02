import os 
import pickle
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import random

def calc_auc(labels, scores):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    return auc

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost

def calc_aupr(labels, scores):
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recall, precision)
    return aupr

def splits(s: str):
    return s.split('#')

def construct_node_name(node):
    # use # as the delimiters
    return "#".join(map(str, node))

def construct_edge_name(node):
    # use # as the delimiters
    return "*".join(map(str, node))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dump(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def calc_norm(x, pow):
    x[x == 0.] = torch.inf
    x = torch.pow(x, pow)
    return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def construct_incidence(association):
    drug_num, disease_num = association.shape
    s1 = np.zeros((drug_num, drug_num))
    s2 = np.zeros((disease_num,disease_num))
    s3 = np.concatenate((s1, association), axis=1)
    s4 = np.concatenate((association.T, s2), axis=1)
    incidence = np.concatenate((s3, s4), axis=0)
    return incidence


def k_fold_split(drdi, seed):
    rows, columns = np.where(drdi>0)
    kflod = KFold(10, shuffle=True, random_state=seed)
    for _, (train_idx, test_idx) in enumerate(kflod.split(range(len(rows)))):
        train_set = np.zeros(drdi.shape, dtype=drdi.dtype)
        test_set = np.zeros(drdi.shape, dtype=drdi.dtype)
        train_set[rows[train_idx], columns[train_idx]] = 1
        test_set[rows[test_idx], columns[test_idx]] = 1
        yield train_set, test_set

        
def fast_in1d(arr1, arr2):
    set_arr2 = set(arr2)
    return np.array([not item in set_arr2 for item in arr1])


def comp(list1, list2):
    for val in list1:
        if val in list2:
            return True
    return False

def bpr_loss_fn(pos_score, neg_score):

    # utilize the broadcast mechanism
    score = pos_score - neg_score
    # loss = torch.abs(-fn.logsigmoid(score) - 0.0009).mean() + 0.0009
    loss = -torch.nn.functional.logsigmoid(score).mean()
    return loss


def weight_bce_loss_fn(pos_score, neg_score):
    pos_score = pos_score.reshape(-1)
    neg_score = neg_score.reshape(-1)
    label_pos = torch.ones_like(pos_score)
    label_neg = torch.zeros_like(neg_score)
    label = torch.concat((label_pos, label_neg))
    predict = torch.concat((pos_score, neg_score))
    predict = torch.nn.functional.sigmoid(predict)
    pos_weight = (neg_score.shape[0] / pos_score.shape[0])
    weight = pos_weight * label + 1 - label
    label = label
    loss = torch.nn.functional.binary_cross_entropy(
        input=predict, target=label, weight=weight)
    return loss

def bce_loss_fn(pos_score, neg_score):
    pos_score = pos_score.reshape(-1)
    neg_score = neg_score.reshape(-1)
    label_pos = torch.ones_like(pos_score)
    label_neg = torch.zeros_like(neg_score)
    label = torch.concat((label_pos, label_neg))
    predict = torch.concat((pos_score, neg_score))
    predict = torch.nn.functional.sigmoid(predict)
    loss = torch.nn.functional.binary_cross_entropy(
        input=predict, target=label)
    return loss 

def calc_contrastive_loss(self, embedding):
    embedding_norm = embedding / \
        torch.norm(embedding, dim=1, p=2, keepdim=True)
    index = torch.where(torch.sum(self.incidence, dim=1) == 0)[0]
    isolated_node = embedding_norm[index]
    index_drug = torch.where(index < self.drug_num)[0]
    index_disease = torch.where(index >= self.drug_num)[0]
    similarity1 = torch.mm(isolated_node, embedding_norm.detach().T)
    # drug contra_loss
    similarity2 = self.drug_similarity[index[index_drug]]
    similarity3 = similarity1[index_drug, :self.drug_num]
    loss1 = torch.mean((similarity2 - similarity3) ** 2)
    # disease contra_loss
    similarity4 = self.disease_similarity[index[index_disease] - self.drug_num]
    similarity5 = similarity1[index_disease, self.drug_num:]
    loss2 = torch.mean((similarity4 - similarity5) ** 2)
    return loss1 + loss2

def calc_contrastive_loss2(self, embedding):
    embedding_norm = embedding / \
        torch.norm(embedding, dim=1, p=2, keepdim=True)

    similarity1 = torch.mm(embedding_norm, embedding_norm.detach().T)
    # drug contra_loss
    similarity2 = self.drug_similarity
    similarity3 = similarity1[:self.drug_num, :self.drug_num]
    loss1 = torch.mean((similarity2 - similarity3) ** 2)
    # drug contra_loss
    similarity4 = self.disease_similarity
    similarity5 = similarity1[self.drug_num:, self.drug_num:]
    loss2 = torch.mean((similarity4 - similarity5) ** 2)
    return loss1 + loss2

def get_valid_set(test_data, drdi):
    drug_num = drdi.shape[0]
    pos_drug, pos_disease = np.where(test_data > 0)
    pos_disease = pos_disease + drug_num
    pos_num = len(pos_drug)
    neg_drug, neg_disease = np.where(drdi ==0)
    negative_sample_index = random.sample(range(len(neg_drug)), pos_num)
    neg_drug = neg_drug[negative_sample_index]
    neg_disease = neg_disease[negative_sample_index] + drug_num
    drug_set = np.concatenate((pos_drug, neg_drug), axis=0)
    disease_set = np.concatenate((pos_disease, neg_disease), axis=0)
    pos_label = [1 for _ in range(pos_num)]
    neg_label = [0 for _ in range(pos_num)]
    label = pos_label + neg_label
    neg_samples = list(zip(neg_drug, neg_disease))
    return torch.tensor(drug_set), torch.tensor(disease_set), label, neg_samples
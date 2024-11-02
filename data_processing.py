import itertools
from scipy import io
from scipy.sparse import csr_matrix
import numpy as np
import os
import torch
from utils import *
import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
import random

class Data_set(Dataset):
    def __init__(self, data, all_data, neg_ratio=10, extra_info=None, mode='bi') -> None:
        super(Data_set, self).__init__()
        self.data = data
        self.all_data = all_data
        self.mode = mode
        self.drug_data_dict = defaultdict(list)
        self.disease_data_dict = defaultdict(list)
        self.data_list = []

        self.neg_ratio = neg_ratio
        self.extra_info = extra_info
        self._processing()

    def _processing(self):
        
        drug_num, disease_num = self.all_data.shape
        self.drug_num, self.disease_num = drug_num, disease_num 
        
        rows, columns = np.where(self.data > 0)
        for row, column in zip(rows, columns):
            self.data_list.append((row, column + drug_num))
        rows, columns = np.where(self.all_data > 0)
        for row, column in zip(rows, columns):
            self.drug_data_dict[row].append(column + drug_num)
            self.disease_data_dict[column + drug_num].append(row)
        if self.extra_info is not None:
            for row, column in self.extra_info:
                self.drug_data_dict[row].append(column)
                self.disease_data_dict[column].append(row)
        self.disease_id = [i + self.drug_num for i in range(self.disease_num)]
        self.drug_id = [i for i in range(self.drug_num)]

    def _generate_negative_samples(self, drug_id, disease_id):
        i = 0
        j = 0
        neg_samples = np.array([], dtype=int)
        
        if self.mode == 'bi':
            size1 = self.neg_ratio // 2
            size2 = size1
        else:
            size1 = self.neg_ratio
            size2 = 0
        while i < size1:
            negative_sample = random.sample(self.disease_id, size1 * 2)
            negative_sample = np.array(negative_sample)
            mask = fast_in1d(negative_sample, self.drug_data_dict[drug_id])

            negative_sample = negative_sample[mask]
            neg_samples = np.concatenate((neg_samples, negative_sample))
            i += negative_sample.size
        neg_samples = neg_samples[:size1]
        while j < size2:
            negative_sample = random.sample(self.drug_id, size2 * 2)
            negative_sample = np.array(negative_sample)
            mask = fast_in1d(negative_sample, self.disease_data_dict[disease_id])

            negative_sample = negative_sample[mask]
            neg_samples = np.concatenate((neg_samples, negative_sample))
            j += negative_sample.size
        neg_samples = neg_samples[:self.neg_ratio]
        return neg_samples
    
    def __len__(self):
        
        return len(self.data_list)
        
    
    def __getitem__(self, index):
        
        drug, diesese = self.data_list[index]
        neg_samples = self._generate_negative_samples(drug, diesese)
        positive_label = [1]
        negative_label = [0 for _ in range(self.neg_ratio)]
        return (torch.tensor([drug]), torch.tensor([diesese]), torch.tensor(neg_samples),
                torch.tensor(positive_label, dtype=torch.float), torch.tensor(negative_label, dtype=torch.float))

if __name__ == '__main__':
    eds = {0: [1, 4, 5], 1: [2, 5, 6], 2: [3, 5],
           3: [1, 4], 4: [1, 2, 3, 5], 5: [2, 6]}
    # a, b, c = get_n_project_adj(eds,4)
    # print(c)
    # print(a)
    # print(b)

    data_path = './data/Cdataset/0'
    

    # g = ConstructNProjectAdj()
    # g.hyperedges_dict = eds
    # a, b, c = get_n_project_adj(g.hyperedges_dict, 3)
    # print(g.hyperedges_dict)
    # print(g.get_degree(2))
    # print(g.get_id2node(2))
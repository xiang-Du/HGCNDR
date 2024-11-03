from data_processing import Data_set
from torch.utils.data import DataLoader
from parse import parse_args
from model import HGCNDR
from scipy import io
from torch.optim import Adam, lr_scheduler
from procedure import train, test
from utils import set_seed, comp, construct_incidence, bpr_loss_fn
import numpy as np
import torch


def process_test_data(test_row, test_data, drug_num):
    drug_id = test_row
    data_len = test_data.shape[0]
    drug_set = np.array([drug_id] * data_len)
    pos_disease = np.where(test_data > 0)[0]
    pos_disease = pos_disease + drug_num
    pos_num = len(pos_disease)
    neg_disease = np.where(test_data ==0)[0]
    neg_num = len(neg_disease)
    neg_disease = neg_disease + drug_num
    disease_set = np.concatenate((pos_disease, neg_disease), axis=0)
    assert drug_set.shape == disease_set.shape
    pos_label = [1 for _ in range(pos_num)]
    neg_label = [0 for _ in range(neg_num)]
    label = pos_label + neg_label
    neg_samples = list(zip(drug_set[:neg_num], neg_disease))
    return torch.tensor(drug_set), torch.tensor(disease_set), label, neg_samples

    
def main():
    args = parse_args()
    epoch = 3000
    seed = args.seed
    mode = args.mode
    neg_ratio = args.neg_ratio
    embed_dim1 = args.embed_dim1
    embed_dim2 = args.embed_dim2
    lr = args.lr
    order = args.order
    layer_num = args.layer_num
    loss_fn = bpr_loss_fn
    rg = args.rg
    k = args.k
    variable_weight = args.variable_weight
    set_seed(seed)
  
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    avg_auc1, avg_aupr1 = 0, 0
    avg_auc2, avg_aupr2 = 0, 0
    # drug_attr = io.loadmat('./data/feature.mat')['feature']\
    data_path = args.data_path
    data = io.loadmat(data_path)
    disease_similarity = data['disease']
    drug_similarity = data['drug']
    didr = data['didr']
    drdi = didr.T
    drug_num = drdi.shape[0]
    for test_row in range(drug_num):
        # if test_row == 0:
        #     continue
        test_data = drdi[test_row]
        train_data = drdi.copy()
        train_data[test_row] = 0
        drug_set, disease_set, label, valid_neg_sample = process_test_data(test_row, test_data, drug_num)
        valid_data_loader = ((drug_set, disease_set, label),)

        dd_train_set = Data_set(train_data, drdi, neg_ratio,
                                extra_info=valid_neg_sample, mode='bi')
        train_data_loader = DataLoader(
            dd_train_set, batch_size=len(dd_train_set), shuffle=True)

        
        # #######################################################
        # flag = None
        # for i in range(10000):
        #     for data in train_data_loader:
        #         h, t, neg_t, pos_label, neg_label = data
        #         print(neg_t[-1, -10:])
        #         n_t = neg_t[:, :10]
        #         n_h = neg_t[:, 10:]
        #         h = h.repeat(1, neg_ratio // 2).contiguous().view(-1).to('cpu').tolist()
        #         n_t = n_t.reshape(-1).to('cpu').tolist()
        #         t = t.repeat(1, neg_ratio // 2).contiguous().view(-1).to('cpu').tolist()
        #         n_h = n_h.reshape(-1).to('cpu').tolist()
        #         train_neg_sample1 = [(h[i], n_t[i]) for i in range(len(h))]
        #         train_neg_sample2 = [(n_h[i], t[i]) for i in range(len(t))]
        #         flag = comp(valid_neg_sample + dd_valid_set.data_list, train_neg_sample1+ train_neg_sample2 + dd_train_set.data_list)
        #         if flag:
        #             break
        # if flag:
        #     break
        # print(flag)
        # break
        # ###########################################################
        incidence = construct_incidence(train_data)
        model = HGCNDR(drug_similarity, disease_similarity, embed_dim1, embed_dim2, incidence, layer_num, k=k, variable_weight=variable_weight, order=order, mode=mode, device=device,score='mlp').to(device)


        optimizer = Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * epoch, 0.6 * epoch, 0.8 * epoch], gamma=0.8)
        best_auc1, best_aupr1, position1 = 0 ,0, 0
        best_auc2, best_aupr2, position2 = 0 ,0, 0
        for i in range(epoch): 
            loss, auc, aupr = train(model, train_data_loader, optimizer,loss_fn, rg, device)
            # scheduler.step()
            if (i+1) % 1 == 0:
                print(loss, auc, aupr)
        
                auc, aupr, _, _ = test(model, valid_data_loader, device)
                if auc > best_auc1:
                    best_auc1 = auc 
                    best_aupr1 = aupr
                    position1 = i
                if aupr > best_aupr2:
                    best_auc2 = auc 
                    best_aupr2 = aupr
                    position2 = i

                print(i+1, ':', auc, aupr)
        print('best1:', position1, ':', best_auc1, best_aupr1)
        print('best2:', position2, ':', best_auc2, best_aupr2)
        avg_auc1 += best_auc1
        avg_aupr1 += best_aupr1
        avg_auc2 += best_auc2
        avg_aupr2 += best_aupr2
    print('auc:', avg_auc1/drug_num, 'aupr:', avg_aupr1/drug_num)
    print('auc:', avg_auc2/drug_num, 'aupr:', avg_aupr2/drug_num)



if __name__ == '__main__':
    main()

from data_processing import Data_set
from torch.utils.data import DataLoader
import torch
from parse import parse_args
from model import HGCNDR
from scipy import io
from torch.optim import Adam
from procedure import train, test
from utils import set_seed, construct_incidence, k_fold_split, bpr_loss_fn, get_valid_set
import numpy as np




def main():
    args = parse_args()
    epoch = 5000
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
    avg_auc, avg_aupr = 0, 0
    # drug_attr = io.loadmat('./data/feature.mat')['feature']\
    data_path = args.data_path
    data = io.loadmat(data_path)
    disease_similarity = data['disease']
    drug_similarity = data['drug']
    didr = data['didr']
    drdi = didr.T
    k_fold = k_fold_split(drdi, seed)
    for train_data, test_data in k_fold:
        # train_data = io.loadmat('./data/Cdataset/0/train_set.mat')['didr'].T
        # test_data = io.loadmat('./data/Cdataset/0/test_set.mat')['didr'].T
        drug_set, disease_set, label, valid_neg_sample = get_valid_set(test_data, drdi)
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
        best_auc, best_aupr, position = 0 ,0, 0
        for i in range(epoch): 
            loss, auc, aupr = train(model, train_data_loader, optimizer,loss_fn, rg, device)
            # scheduler.step()
            if (i+1) % 1 == 0:
                print(loss, auc, aupr)
        
                auc, aupr, _, _ = test(model, valid_data_loader, device)
                if auc > best_auc:
                    best_auc = auc 
                    best_aupr = aupr
                    position = i

                print(i+1, ':', auc, aupr)
        print('best:', position, ':', best_auc, best_aupr)
        avg_auc += best_auc
        avg_aupr += best_aupr
    print('auc:', avg_auc/10, 'aupr:', avg_aupr/10)



if __name__ == '__main__':
    main()

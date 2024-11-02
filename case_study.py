from data_processing import Data_set
from torch.utils.data import DataLoader
from parse import parse_args
from model import HGCNDR
from scipy import io
from torch.optim import Adam
from procedure import train, test, case
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
    device = 'cuda'
    avg_auc, avg_aupr = 0, 0
    # drug_attr = io.loadmat('./data/feature.mat')['feature']\
    data_path = './data/Gdataset.mat'
    data = io.loadmat(data_path)
    disease_similarity = data['disease']
    drug_similarity = data['drug']
    didr = data['didr']
    drdi = didr.T
    drug_num, disease_num = drdi.shape
    k_fold = k_fold_split(drdi, seed)
    for _, test_data in k_fold:
        # train_data = io.loadmat('./data/Cdataset/0/train_set.mat')['didr'].T
        # test_data = io.loadmat('./data/Cdataset/0/test_set.mat')['didr'].T
        train_data = drdi
        drug_set, disease_set, label, _ = get_valid_set(test_data, drdi)
        valid_data_loader = ((drug_set, disease_set, label),)

        dd_train_set = Data_set(train_data, drdi, neg_ratio,
                                extra_info=None, mode='bi')
        train_data_loader = DataLoader(
            dd_train_set, batch_size=len(dd_train_set), shuffle=True)
        incidence = construct_incidence(train_data)
        model = HGCNDR(drug_similarity, disease_similarity, embed_dim1, embed_dim2, incidence, layer_num, k=k, variable_weight=variable_weight, order=order, mode=mode, device='cuda',score='mlp').to(device)

        optimizer = Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[0.3 * epoch, 0.6 * epoch, 0.8 * epoch], gamma=0.8)
        best_auc, best_aupr, position = 0 ,0, 0
        for i in range(epoch): 
            loss, auc, aupr = train(model, train_data_loader, optimizer,loss_fn, rg, device)
            # scheduler.step()
            if (i+1) % 10 == 0:
                print(loss, auc, aupr)
        
                auc, aupr, _, _ = test(model, valid_data_loader, device)
                if auc > best_auc:
                    best_auc = auc 
                    best_aupr = aupr
                    position = i

                print(i+1, ':', auc, aupr)
            if (i+1) % 10 == 0:
                file = './final/case_study/case_study_'+str(i+1) +'.npy'
                score = case(model,drug_num, disease_num, device)
                np.save(file, score)
                
        print('best:', position, ':', best_auc, best_aupr)
        avg_auc += best_auc
        avg_aupr += best_aupr
        break



if __name__ == '__main__':
    main()

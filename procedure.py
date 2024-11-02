import torch
from utils import calc_auc, calc_aupr, common_loss
from model import HGCNDR


def train(model:HGCNDR, train_dataloader, optimizer, loss_fn, rg, device):
    model.train()
    avg_loss = 0
    all_logit = []
    all_label = []
    size = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        h, t, neg_t, pos_label, neg_label = data
        h = h.to(device)
        t = t.to(device)
        neg_t = neg_t.to(device)
        pos_score, neg_score, feature_s, feature_h = model(h, t, neg_t)
        loss_com_drug = common_loss(feature_s[:model.drug_num], feature_h[:model.drug_num])
        loss_com_dis = common_loss(feature_s[model.drug_num:], feature_h[model.drug_num:])
        loss = loss_fn(pos_score, neg_score) + rg * (loss_com_dis + loss_com_drug)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.detach().to('cpu').item()
        ########################################
        score = torch.concat((pos_score.contiguous().view(-1),
                          neg_score.contiguous().view(-1)))
        all_logit = all_logit + score.to('cpu').detach().tolist()
        label = torch.concat((pos_label.view(-1).to('cpu'), neg_label.view(-1).to('cpu')))
        all_label = all_label + label.tolist()
    auc = calc_auc(all_label, all_logit)
    aupr = calc_aupr(all_label, all_logit)
    return avg_loss / size, auc, aupr


def test(model, valid_dataloader, device):
    model.eval()
    all_logit = []
    all_label = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            h, t, label = data
            h = h.to(device)
            t = t.to(device)
            
            score = model.predict(h, t)
            all_logit = all_logit + score.to('cpu').detach().tolist()
            
            all_label = all_label + label
    auc = calc_auc(all_label, all_logit)
    aupr = calc_aupr(all_label, all_logit)
    return auc, aupr, all_logit, all_label

def case(model, drug_num, disease_num, device):
    model.eval()
    drug = torch.tensor([[i]*disease_num for i in range(drug_num)])
    drug = drug.reshape(-1)
    disease = torch.tensor([list(range(drug_num, drug_num + disease_num))*drug_num])
    disease = disease.reshape(-1)



    with torch.no_grad():

        h = drug.to(device)
        t = disease.to(device)
        
        score = model.predict(h, t)
        score = score.to('cpu').detach().reshape(drug_num, disease_num)
        score = score.numpy()
        
    
    return score
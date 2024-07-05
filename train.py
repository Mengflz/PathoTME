import os
import glob
import numpy as np
import torch
import torch.nn as nn
import statistics
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
import yaml

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from code.Datasets import MultiomicsDataset, stratified_dataset
from code.model import *
from code.metrics_utils import *
from code.utils import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def SNN_training(train_loader, input_dim, size, n_classes):
    model_SNN = SNN_token(input_dim, size, n_classes, cls_flg=True)
    model_SNN.to('cuda')
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_SNN.parameters()), lr=2e-4, weight_decay=1e-5) # parameter from MCAT

    print('Gene guide branch training...')
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        model_SNN.train()
        for batch_idx,(_, gene, label, _) in enumerate(train_loader):
            gene = gene.to(device='cuda')
            label = label.to(device='cuda')

            inputs = gene.float()
            optimizer.zero_grad()

            logits,prob,_ = model_SNN(inputs)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    model_SNN.eval()
    model = model_SNN.fc_omic
    return model

def train(dataset, cfg, epochs=100, kfolds=5):
    out_model_dir = os.path.join(cfg['result_dir'], f'{kfolds}folds/')
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
    
    size = cfg['model_size_omic']
    size_dict = {'small': 256, 'medium': 512, 'big': 1024}
    cancer_type = cfg['cancer_type']
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    patience = cfg['patience']
    stop_epoch = cfg['stop_epoch']
    trained_SNN = cfg['trained_SNN']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg['seed'])

    criterion_sim = nn.CosineSimilarity(dim=1)
    criterion_label = nn.CrossEntropyLoss()
    criterion_D = nn.CrossEntropyLoss()
    
    input_dim = dataset.feature_dim

    for i, (train_index, val_index) in enumerate(skf.split(dataset, dataset.tissue_num)):
        print(f"Fold {i}")

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=10)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=10)

        if trained_SNN:
            model_SNN = SNN_training(train_loader, input_dim, size, n_classes=4)
        else:
            model_SNN = SNN_token(input_dim, size, n_classes=4)

        model_WSI = ABMIL_VPT(cfg['prompt_length'], L=size_dict[size])
        model_cls = cls_wsi(size_dict[size], n_classes=4)
        model_D = Discriminator(size_dict[size], cancer_type)
        
        optimizer_wsi = torch.optim.Adam(filter(lambda p: p.requires_grad, model_WSI.parameters()), lr=lr, weight_decay=weight_decay)
        optimizer_cls = torch.optim.Adam(filter(lambda p: p.requires_grad, model_cls.parameters()), lr=lr, weight_decay=weight_decay)
        optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, model_D.parameters()), lr=lr, weight_decay=weight_decay)
        
        writer = SummaryWriter(os.path.join(out_model_dir, f'runs/Kfold_{i}'))
        out_model_path_wsi = os.path.join(out_model_dir, f'WSI_Kfold_{i}.pt')
        out_model_path_snn = os.path.join(out_model_dir, f'SNN_Kfold_{i}.pt')
        out_model_path_cls = os.path.join(out_model_dir, f'CLS_Kfold_{i}.pt')
        out_model_path_D = os.path.join(out_model_dir, f'D_Kfold_{i}.pt')

        early_stoppers = [EarlyStopping(patience, stop_epoch, verbose=(i==0)) for i in range(4)]
        
        print(f"Fold {i}: WSI learning branch training...")
        for epoch in range(epochs):
            train_loss = 0.0
            D_loss = 0.0
            model_WSI.train()
            model_WSI.to('cuda')
            model_cls.train()
            model_cls.to('cuda')
            model_D.train()
            model_D.to('cuda')
            
            for batch_idx,(wsi, gene, label, tissue_num) in enumerate(train_loader):
                gene = gene.to('cuda')
                label = label.to('cuda')
                wsi = wsi.to('cuda')
                tissue_num = tissue_num.to('cuda')
                wsi = wsi.squeeze()
                
                # DANN 
                token_wsi, _, _, _ = model_WSI(wsi)
                p_tissue = model_D(token_wsi.detach())
                loss_D = criterion_D(p_tissue, tissue_num)
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                
                # CLS
                token_gene = model_SNN(gene.float()) 
                logits, prob, Y_hat = model_cls(token_wsi)
                p_tissue = model_D(token_wsi)
                
                loss_1 = 1-criterion_sim(token_wsi, token_gene)
                loss_2 = criterion_label(logits, label)
                loss_D = criterion_D(p_tissue, tissue_num)
                
                lamda = 0.1*get_lambda(epoch, epochs)
                
                loss = (1-lamda)*loss_1 + lamda*loss_2 - lamda*loss_D
                
                optimizer_wsi.zero_grad()
                optimizer_cls.zero_grad()
                optimizer_D.zero_grad()
                loss.backward()
                optimizer_wsi.step()
                optimizer_cls.step()
                
                train_loss += loss.item()
                D_loss += loss_D.item()
                
            # evaluate on validation set
            model_WSI.eval()
            model_SNN.eval()
            model_cls.eval()
            model_D.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                predicted_labels = []; labels = []; probs = []; corrects = torch.zeros(1).to('cuda')
                for batch_idx, (wsi, gene, label, tissue_num) in enumerate(val_loader):
                    gene = gene.to('cuda')
                    label = label.to('cuda')
                    wsi = wsi.to('cuda')
                    tissue_num = tissue_num.to('cuda')
                    wsi = wsi.squeeze()
                    
                    token_wsi, _, _, _ = model_WSI(wsi)
                    token_gene = model_SNN(gene.float()) # 1 x 1024
                    logits, prob, Y_hat = model_cls(token_wsi)
                    p_tissue = model_D(token_wsi.detach())

                    loss_1 = 1-criterion_sim(token_wsi, token_gene)
                    loss_2 = criterion_label(logits, label)
                    loss_D = criterion_D(p_tissue, tissue_num)
                    lamda = 0.1*get_lambda(epoch, epochs)
                    
                    corrects += (tissue_num==torch.argmax(p_tissue).item()).sum()
                    acc_D = corrects.item() / len(val_loader)
                    loss = (1-lamda)*loss_1 + lamda*loss_2 - lamda*loss_D
                    # print(f'---{batch_idx}:loss{loss}; similarity{criterion_sim(token_wsi, token_gene)}')
                    
                    predicted_labels.extend(Y_hat.detach().cpu())
                    labels.extend(label.cpu())
                    probs.extend(prob.cpu())
                    val_loss += loss.item() 
                        

                accuracy, precision, recall, f1, roc_auc, pr_auc = cal_metrics(labels, predicted_labels, probs)

                print(f'Epoch {epoch + 1}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {accuracy}, ROCAUC:{roc_auc:.4f}, PR_AUC:{pr_auc:.4f}')
                print(f'Epoch {epoch + 1}, Discriminator Accuracy: {acc_D}, Discriminator Loss: {D_loss/len(val_loader)}')
                print(f'Epoch {epoch + 1}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
                
                writer.add_scalar('Train Loss', float(train_loss), epoch)
                writer.add_scalar('Val Loss', float(val_loss), epoch)
                writer.add_scalar('ROCAUC', float(roc_auc), epoch)
                writer.add_scalar('Discriminator Accuracy', float(acc_D), epoch)
                writer.add_scalar('Discriminator Loss', float(D_loss/len(val_loader)), epoch)

                writer.add_scalar('Accuracy', float(accuracy), epoch)
                writer.add_scalar('Precision', float(precision), epoch)
                writer.add_scalar('Recall', float(recall), epoch)
                writer.add_scalar('F1 score', float(f1), epoch)

                early_stoppers[0](epoch, val_loss/len(val_loader), model_WSI, ckpt_name=out_model_path_wsi)
                early_stoppers[1](epoch, val_loss/len(val_loader), model_SNN, ckpt_name=out_model_path_snn)
                early_stoppers[2](epoch, val_loss/len(val_loader), model_cls, ckpt_name=out_model_path_cls)
                early_stoppers[3](epoch, val_loss/len(val_loader), model_D, ckpt_name=out_model_path_D)

                if early_stoppers[0].early_stop:
                    print("Early stopping")
                    break


def test(custom_dataset, cfg):
    out_model_dir = cfg['result_dir']
    size = cfg['model_size_omic']
    size_dict = {'small': 256, 'medium': 512, 'big': 1024}

    input_dim = custom_dataset.feature_dim
    criterion = nn.CrossEntropyLoss()
    roc_list = []; acc_list=[]; f1_list = []; pr_auc_list=[]
    for i in range(5):
        kfold = i
        model_wsi_path = os.path.join(out_model_dir,'5folds', f'WSI_Kfold_{i}.pt')
        model_cls_path = os.path.join(out_model_dir,'5folds', f'CLS_Kfold_{i}.pt')
        
        model_WSI = ABMIL_VPT(cfg['prompt_length'], L=size_dict[size])
        model_cls = cls_wsi(size_dict[size], n_classes=4)

        model_WSI.load_state_dict(torch.load(model_wsi_path), strict=True)
        model_WSI.to('cuda')
        model_cls.load_state_dict(torch.load(model_cls_path), strict=True)
        model_cls.to('cuda')

        model_WSI.eval()
        model_cls.eval()

        test_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True, num_workers=0)
        test_loss = 0.0
        with torch.no_grad():
            predicted_labels = []; labels = []; probs = []
            for batch_idx, (wsi, gene, label, tissue_num) in enumerate(test_loader):
                label = label.to('cuda')
                wsi = wsi.to('cuda')
                gene = gene.to('cuda').float()

                inputs = wsi.squeeze()
                token_wsi, _, _, _ = model_WSI(inputs)
                
                logits, prob, Y_hat = model_cls(token_wsi)
                
                predicted_labels.extend(Y_hat.detach().cpu())
                labels.extend(label.cpu())
                probs.extend(prob.cpu())
                test_loss += criterion(logits, label)
                
            test_loss = test_loss/len(test_loader)
            accuracy, precision, recall, f1, roc_auc, pr_auc = cal_metrics(labels, predicted_labels, probs, 
                                                                    result_dir=out_model_dir,
                                                                    kfold=str(kfold), save_csv=True)
            
            roc_list.append(roc_auc); f1_list.append(f1); acc_list.append(accuracy); pr_auc_list.append(pr_auc)
            # accuracy_total = correct / total
            print(f'{kfold} Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}, ROCAUC:{roc_auc:.4f} ')
            print(f'{kfold} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROCAUC:{roc_auc:.4f} ')
    print(f'Mean: Accuracy: {sum(acc_list)/5:.4f}, F1: {sum(f1_list)/5:.4f}, ROCAUC:{sum(roc_list)/5:.4f}, PR_AUC:{sum(pr_auc_list)/5:.4f}')
    print(f'STD_ROC:{statistics.stdev(roc_list):.4f}, STD_Acc:{statistics.stdev(acc_list):.4f}, STD_F1:{statistics.stdev(f1_list):.4f}, STD_PR_AUC:{statistics.stdev(pr_auc_list):.4f}')


def load_dataset(cfg):
    custom_dataset = MultiomicsDataset(label_df_path='data/label_id.csv', 
                                   mode='HIPT', wsi = True, scaler=False, gene=True,
                                   gene_file_path = 'data/sample_knowledge_exp.csv')
    
    train_mask, _, test_mask = stratified_dataset(custom_dataset, custom_dataset.tissue_num, test_size=0.15, val_size=0, seed=3076) 
    
    train_set = MultiomicsDataset(label_df_path='data/label_id.csv', indice=train_mask,
                                   mode='HIPT', wsi = True, scaler=False,gene = True,
                                   gene_file_path = 'data/sample_knowledge_exp.csv')
    test_set = MultiomicsDataset(label_df_path='data/label_id.csv', indice=test_mask,
                                   mode='HIPT', wsi = True, scaler=False,gene = True,
                                   gene_file_path = 'data/sample_knowledge_exp.csv')
                            
    return train_set, test_set


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    train_set, test_set = load_dataset(cfg['datasets'])
    print(f'Train set: {len(train_set)}, Test set: {len(test_set)} samples')
    
    # train(train_set, cfg['models'])
    test(test_set, cfg['models'])
    print('PathoTME Done!')

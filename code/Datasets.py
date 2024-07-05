import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore
import pickle


class MultiomicsDataset(Dataset):
    '''Return (wsi, gene, label) for each row in dataframe from label_df_path'''
    def __init__(self, label_df_path, gene_file_path, 
                 indice=None, wsi = False, gene = True, mode='HIPT',
                 scaler=False, label_onehot=False, **kwargs):
        self.gene_flg = gene
        self.wsi_flg = wsi
        self.scaler = scaler
        self.label_onehot = label_onehot
        self.mode = mode

        self.info_data = pd.read_csv(label_df_path)

        if indice:
            self.info_data = self.info_data.iloc[indice,:].reset_index()
        self.label = self.info_data['MFP'].replace({'IE': 0, 'F': 1, 'D': 2, 'IE/F': 3})
        self.expression_id = self.info_data['sample_id']
        if mode == 'HIPT': 
            # self.wsi_token_path = 'data/HIPT_token'
            with open('data/sample_HIPT_features.pkl', 'rb') as f:
                self.wsi_total = pickle.load(f)
        elif mode == 'CTranspath':
            self.wsi_token_path = 'data/CTranspath_features.pkl'
        else:
            raise ValueError('mode not implemented')


        self.tissue = self.info_data['HISTOLOGICAL_SUBTYPE']
        # self.tissue_num = self.tissue.replace({'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC_AC': 3, 'CESC_SCC': 4, 'CHOL': 5, 'COREAD': 6,
        #                                        'ESCA_SCC': 7, 'ESGA_AC': 8, 'HNSC': 9, 'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LICH': 13, 
        #                                        'LUAD': 14, 'LUSC': 15, 'OV': 16, 'PAAD': 17, 'PRAD': 18, 'SKCM': 19, 'THCA': 20, 
        #                                        'UCEC': 21, 'UVM': 22})
        label_encoder = LabelEncoder()
        self.tissue_num = label_encoder.fit_transform(self.tissue)
        
        if gene:
            self.expression = pd.read_csv(gene_file_path, index_col=0)
            if indice:
                self.expression = self.expression.iloc[indice,:]
            self.expression = self.expression.to_numpy()
            self.feature_dim = self.expression.shape[1]
            if scaler:
                self.expression = zscore(self.expression, axis=1)                        
    
    def __len__(self):
        return self.info_data.shape[0]
    
    def __getitem__(self, idx):
        wsi = np.nan ; gene = np.nan
        label = self.label[idx]
        tissue = self.tissue_num[idx]
        if self.wsi_flg:
            self.wsi_path = self.info_data.iloc[idx]['wsi_name']
            wsi = self.wsi_total[self.wsi_path]
        if self.gene_flg:
            gene = self.expression[idx]
        if self.label_onehot:
            label = np.eye(4)[label]
            
        return (wsi, gene, label, tissue)


def stratified_dataset(dataset, labels, test_size=0.15, val_size=0.15, seed=8, **kwargs):
    '''Use this function to stratify the dataset into train, validation and test sets
    return index of train, validation and test sets'''
    len_dataset = len(dataset)
    train_mask,test_mask = train_test_split(range(len_dataset), test_size=test_size, random_state=seed, stratify=labels)
    val_size = val_size/(1-test_size)
    if val_size:
        train_mask, val_mask = train_test_split(train_mask, test_size=val_size, random_state=seed, stratify=labels[train_mask])
    else: val_mask = None
    return train_mask, val_mask, test_mask


def split_datasets(dataset, labels, test_size=0.2, val_size=0, seed=8):
    train_mask, val_mask, test_mask = stratified_dataset(dataset, labels, test_size, val_size, seed)
    train_dataset = Subset(dataset, train_mask)
    if val_mask:
        validate_dataset = Subset(dataset, val_mask)
    else: validate_dataset = None
    test_dataset = Subset(dataset, test_mask)
    return train_dataset, validate_dataset, test_dataset

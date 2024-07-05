import numpy as np
import pandas as pd
from numpy import vstack
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,precision_recall_fscore_support,average_precision_score
from sklearn.preprocessing import label_binarize


def class_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy:{acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    class_acc = {}
    for i in range(len(cm)):
        class_acc[i] = cm[i, i] / cm[i, :].sum()

    return class_acc

def optimal_thresh(fpr, tpr, thresholds, p=0):
   loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
   idx = np.argmin(loss, axis=0)
   return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions,result_dir=None, kfold=None, save_csv=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)

    # Save metrics to CSV
    metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': fscore,
    'ROC AUC': auc_value
    }

    metrics_df = pd.DataFrame([metrics_dict])
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(folder_dir):
    #     os.makedirs(folder_dir)
    if save_csv:
        if not os.path.exists(result_dir):
            if kfold:
                result_dir = f'{result_dir}/metrics/'
            os.makedirs(result_dir)

        if kfold:
            metrics_filename = f'{result_dir}/fold{kfold}.csv'
        else:
            metrics_filename = f'{result_dir}/metrics.csv'
            
        metrics_df.to_csv(metrics_filename, index=False)
    return accuracy, auc_value, precision, recall, fscore

def roc_threshold(label, prediction):
   fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
   fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
   c_auc = roc_auc_score(label, prediction)
   return c_auc, threshold_optimal

def cal_metrics(true_labels, predicted_labels, pred_scores, result_dir=None, kfold=None, save_csv=False, 
                three_label=False, verbose=False):
    '''Calculate metrics for classification tasks
    return accuracy, precision, recall, f1, macro_roc_auc_ovr, pr_auc'''
    # # Compute ROC curve and AUC
    # fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    # roc_auc, optim_thresh = roc_threshold(true_labels, pred_scores)

    # predicted_labels = (pred_scores > optim_thresh).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    accuracy_dict = class_accuracy(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division="warn", average='macro')
    recall = recall_score(true_labels, predicted_labels, zero_division="warn", average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    if three_label:
        labels_one_hot = label_binarize(true_labels, classes=[0,1,2])
    else:
        labels_one_hot = label_binarize(true_labels, classes=[0,1,2,3])
    pred_scores = vstack(pred_scores)
    # print(pred_scores, pred_scores.shape)
    try:
        macro_roc_auc_ovr = roc_auc_score(labels_one_hot, pred_scores, multi_class="ovr", average="macro",)
        pr_auc = average_precision_score(labels_one_hot, pred_scores, average="macro")
        # calculate roc auc for each category
        rocauc_per_class = []
        for class_index in range(pred_scores.shape[1]):
            roc_auc = roc_auc_score(labels_one_hot[:, class_index], pred_scores[:, class_index])
            rocauc_per_class.append(roc_auc)
        if verbose:
            for class_index, roc_auc in enumerate(rocauc_per_class):
                print(f"ROC AUC of Class {class_index:.4f} is: {roc_auc:.4f}")
    except ValueError:
        macro_roc_auc_ovr = 0
        pr_auc = 0
        rocauc_per_class = [0,0,0,0]
    
    # Save metrics to CSV
    metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': macro_roc_auc_ovr,
    'PR AUC': pr_auc
    }
    metrics_dict.update({f'{item} Accuracy': accuracy_dict[item] for item in accuracy_dict.keys()})
    metrics_dict.update({f'{item} rocauc': rocauc_per_class[item] for item in range(pred_scores.shape[1])})

    metrics_df = pd.DataFrame([metrics_dict])
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(folder_dir):
    #     os.makedirs(folder_dir)
    if save_csv:
        if not os.path.exists(result_dir):
            if kfold:
                result_dir = f'{result_dir}/metrics/'
            os.makedirs(result_dir)

        if kfold:
            metrics_filename = f'{result_dir}/fold{kfold}.csv'
        else:
            metrics_filename = f'{result_dir}/metrics.csv'
            
        metrics_df.to_csv(metrics_filename, index=False)
    
    return accuracy, precision, recall, f1, macro_roc_auc_ovr, pr_auc

def cal_metrics_ori(true_labels, predicted_labels, pred_scores, result_dir=None, kfold=None, save_csv=False, three_label=False):

    # # Compute ROC curve and AUC
    # fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    # roc_auc, optim_thresh = roc_threshold(true_labels, pred_scores)

    # predicted_labels = (pred_scores > optim_thresh).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    accuracy_dict = class_accuracy(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division="warn", average='macro')
    recall = recall_score(true_labels, predicted_labels, zero_division="warn", average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    if three_label:
        labels_one_hot = label_binarize(true_labels, classes=[0,1,2])
    else:
        labels_one_hot = label_binarize(true_labels, classes=[0,1,2,3])
    pred_scores = vstack(pred_scores)
    macro_roc_auc_ovr = roc_auc_score(labels_one_hot, pred_scores, multi_class="ovr", average="macro",)
    
    # Save metrics to CSV
    metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': macro_roc_auc_ovr
    }
    metrics_dict.update({f'{item} Accuracy': accuracy_dict[item] for item in accuracy_dict.keys()})

    metrics_df = pd.DataFrame([metrics_dict])
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(folder_dir):
    #     os.makedirs(folder_dir)
    if save_csv:
        if not os.path.exists(result_dir):
            if kfold:
                result_dir = f'{result_dir}/metrics/'
            os.makedirs(result_dir)

        if kfold:
            metrics_filename = f'{result_dir}/fold{kfold}.csv'
        else:
            metrics_filename = f'{result_dir}/metrics.csv'
            
        metrics_df.to_csv(metrics_filename, index=False)
    
    return accuracy, precision, recall, f1, macro_roc_auc_ovr
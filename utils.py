# -*- coding: utf-8 -*-
"""
Create Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Load multiple numpy arrays
def load_np(data_folder, file_name_list):
    files = {}
    for file_name in file_name_list:
        files[file_name] = np.load(str(data_folder)+str(file_name)+'.npy', allow_pickle=True)
    return files

# Save multiple numpy arrays
def save_np(data_folder, file_name_dict):
    for k,v in zip(file_name_dict.keys(),file_name_dict.values()):
        np.save(str(data_folder) + str(k)+ '.npy', v)


# Confusion Matrix: to display true positive, true negative, 
#     false positive, and false negative values for each model
def disp_confmat(y_true, y_pred):
    confmat = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = sns.heatmap(confmat, cmap='Oranges', annot=True, fmt="d")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])
    ax.set_title(r"Confusion matrix",fontsize=12)
    plt.show()
    return confmat


# Method for plotting ROC curve: to compare among models in term of the discrimination
def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='AUC = %0.4f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, auc


# Method for plotting PR Curve: to compare among models in term of the precision and recall
def plot_pr_curve(y_true, y_prob):
    pr, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    avg_precision = average_precision_score(y_true, y_prob, pos_label=1)
    plt.figure()
    plt.plot(recall, pr, color='seagreen',
             label='AUC = %0.4f' % avg_precision)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
    return pr, recall, avg_precision

# Method for model evaluation
def evaluate_model(y_true, y_pred, y_prob):
    y_true = y_true
    y_pred = y_pred
    y_prob = y_prob
    confmat = disp_confmat(y_true, y_pred)
    fpr, tpr, auc = plot_roc(y_true, y_prob)
    pr, recall, avg_precision = plot_pr_curve(y_true, y_prob)
    return confmat, fpr, tpr, auc, pr, recall, avg_precision

# -*- coding: utf-8 -*-
"""
Create Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from itertools import cycle

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

# Confusion Matrix: to display number of true positive, true negative, 
#     false positive, and false negative values for each model
def disp_confmat(y_true, y_pred, list_classes):
    confmat = confusion_matrix(y_true, y_pred, labels=list_classes)
    ax = sns.heatmap(confmat, cmap='Oranges', annot=True, fmt="d")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.xaxis.set_ticklabels(list_classes)
    ax.yaxis.set_ticklabels(list_classes)
    ax.set_title(r"Confusion matrix",fontsize=12)
    plt.show()
    return confmat

# Method for plotting ROC curve for muliclassification: 
# to compare among models in term of the discrimination
def plot_roc_multiclass(y_true, y_prob, n_classes):
    
    # 1. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # 2. Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute the AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'salmon'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic of multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, roc_auc


# Method for plotting PR Curve: to compare among models in term of the precision and recall
def plot_pr_multiclass(y_true, y_prob, n_classes):
    
    # 1. Compute PR curve and PR area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    
    # 2. Compute micro-average PR curve and PR area
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_prob, average="micro")  
    
    # Plot all PR curves
    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.4f})'
                  ''.format(average_precision["micro"]))
    
    colors = cycle(['limegreen','blueviolet','hotpink','darkorange'])
    for i, color in zip(range(n_classes), colors):
      l, = plt.plot(recall[i], precision[i], color=color, lw=2)
      lines.append(l)
      labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                    ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve of multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    return precision, recall, average_precision


# Method for model evaluation for multiclassification
def evaluate_model(y_true, y_pred, y_prob, list_classes, n_classes):
    y_true = y_true
    y_pred = y_pred
    y_prob = y_prob
    n_classes = n_classes
    list_classes = list_classes
    confmat = disp_confmat(y_true, y_pred, list_classes)
    fpr, tpr, auc = plot_roc_multiclass(y_true, y_prob, n_classes)
    pr, recall, avg_precision = plot_pr_multiclass(y_true, y_prob, n_classes)
    return confmat, fpr, tpr, auc, pr, recall, avg_precision

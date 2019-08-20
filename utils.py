# -*- coding: utf-8 -*-
"""
Create Methods
"""
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, multilabel_confusion_matrix
from itertools import cycle
from scipy import interp


# Confusion Matrix: to display number of true positive, true negative, 
#     false positive, and false negative values for each model
def plot_conf_multiclass(y_true, y_pred, list_classes):
    fig, axes = plt.subplots(int(np.ceil(len(list_classes) / 2)), 2, figsize=(10, 10))
    axes = axes.flatten()
    multiclass_cm = multilabel_confusion_matrix(y_true, y_pred)
    for i, conf in enumerate(multiclass_cm):
        tn, fp, fn, tp = conf.ravel()
        f1 = 2 * tp / (2 * tp + fp + fn + sys.float_info.epsilon)
        recall = tp / (tp + fn + sys.float_info.epsilon)
        precision = tp / (tp + fp + sys.float_info.epsilon)
        conf_mat = np.array([[tp,fn],
                             [fp,tn]])
        
        # Define color of confusion matrix
        ax = axes[i]
        ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Oranges)
        
        # Define color of annotation
        for a, b in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            ax.text(b, a, conf_mat[a, b],
                    horizontalalignment="center",
                    color="white" if conf_mat[a, b] > conf_mat.max() / 2. else "black")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks), ax.xaxis.set_ticklabels(['1','0'])
        ax.set_yticks(tick_marks), ax.yaxis.set_ticklabels(['1','0'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')    
        ax.set_title('Label: {}\nf1={:.5f}\nrecall={:.5f}\nprecision={:.5f}'.format(list_classes[i], f1, recall, precision))
        ax.grid(False)
        plt.tight_layout()
    
    return multiclass_cm


# Method for plotting ROC curve for muliclassification: 
# to compare among models in term of the discrimination
def plot_roc_multiclass(y_true, y_prob, n_classes):
    
    # Compute ROC curve, ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve and ROC area
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
             label='macro-average ROC curve (area = {roc_auc["macro"]:0.4f})',
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'salmon'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr, tpr, roc_auc


# Method for plotting PR Curve: to compare among models in term of the precision and recall
def plot_pr_multiclass(y_true, y_prob, n_classes):
    
    # Compute PR curve, and PR area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        
    # Compute macro-average PR curve and PR area
    # A "macro-average": quantifying score on average
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision["macro"] = average_precision_score(y_true, y_prob, average="macro")
    
    
    # Plot all PR curves
    
    lines = []
    labels = []
    
    l, = plt.plot(recall["macro"], precision["macro"], color='gold', lw=2)
    lines.append(l)
    labels.append('macro-average Precision-recall (area = {average_precision["macro"]:0.4f})')
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'salmon'])
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
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
    return precision, recall, average_precision


# Method for model evaluation for multiclassification
def evaluate_model(y_true, y_pred, y_prob, list_classes, n_classes):
    y_true = y_true
    y_pred = y_pred
    y_prob = y_prob
    n_classes = n_classes
    list_classes = list_classes
    confmat = plot_conf_multiclass(y_true, y_pred, list_classes)
    fpr, tpr, roc_auc = plot_roc_multiclass(y_true, y_prob, n_classes)
    precision, recall, average_precision = plot_pr_multiclass(y_true, y_prob, n_classes)
    return confmat, fpr, tpr, roc_auc, precision, recall, average_precision

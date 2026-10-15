"""Metricas de discriminacao usadas em credito: AUC, KS e Gini."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y, score):
    return float(roc_auc_score(y, score))


def ks(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def gini(y, score):
    return 2.0 * auc(y, score) - 1.0


def resumo(y, score):
    return {"auc": auc(y, score), "ks": ks(y, score), "gini": gini(y, score)}


from __future__ import annotations
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_confusion(cm, classes, out_base: Path):
    cmn = cm.astype(float)/(cm.sum(axis=1, keepdims=True)+1e-12)
    plt.figure(figsize=(7,6)); sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (normalized)"); plt.xlabel("Pred"); plt.ylabel("True")
    out_base.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(out_base.with_suffix(".png")); plt.savefig(out_base.with_suffix(".svg")); plt.close()

def plot_roc_ovr(y_true, y_proba, classes, out_base: Path):
    y_bin = label_binarize(y_true, classes=list(range(len(classes)))); plt.figure(figsize=(7,6))
    for i in range(len(classes)):
        fpr,tpr,_ = roc_curve(y_bin[:,i], y_proba[:,i]); plt.plot(fpr,tpr,label=f"{classes[i]}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC OvR"); plt.legend()
    out_base.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(out_base.with_suffix(".png")); plt.savefig(out_base.with_suffix(".svg")); plt.close()

def plot_reliability(y_true, y_proba, out_base: Path, n_bins=10):
    conf = y_proba.max(axis=1); pred = y_proba.argmax(axis=1); acc = (pred==y_true).astype(float)
    bins = np.linspace(0,1,n_bins+1); mids, accs = [], []
    for i in range(n_bins):
        m,M=bins[i],bins[i+1]; mask=(conf>=m)&(conf<M if i<n_bins-1 else conf<=M)
        if not np.any(mask): continue
        mids.append((m+M)/2); accs.append(acc[mask].mean())
    plt.figure(figsize=(6,4)); plt.plot([0,1],[0,1],"--"); 
    if mids: plt.plot(mids, accs, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability Diagram")
    out_base.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(out_base.with_suffix(".png")); plt.savefig(out_base.with_suffix(".svg")); plt.close()

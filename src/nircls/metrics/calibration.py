
import numpy as np
def expected_calibration_error(y_true, y_proba, n_bins=15):
    y_true = np.asarray(y_true); conf = y_proba.max(axis=1); preds = y_proba.argmax(axis=1); acc = (preds==y_true).astype(float)
    bins = np.linspace(0,1,n_bins+1); ece=0.0
    for i in range(n_bins):
        m,M=bins[i],bins[i+1]; mask=(conf>=m)&(conf<M if i<n_bins-1 else conf<=M)
        if not np.any(mask): continue
        ece += abs(acc[mask].mean() - conf[mask].mean()) * (mask.mean())
    return float(ece)

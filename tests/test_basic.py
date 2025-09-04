
import numpy as np
from sklearn.pipeline import Pipeline
from nircls.preprocessors.spectro import SNV, SavitzkyGolayTransformer
from nircls.models.factory import PLSDA

def test_pipeline_fit_predict():
    X = np.random.RandomState(0).randn(30, 60); y = (np.arange(30)%2).astype(int)
    pipe = Pipeline([("snv", SNV()), ("sg", SavitzkyGolayTransformer(15,2,1)), ("clf", PLSDA(n_components=3))])
    pipe.fit(X[:20], y[:20]); pipe.predict(X[20:])

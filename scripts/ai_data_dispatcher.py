from joblib import Parallel, delayed, dump, load
import pandas as pd
import numpy as np
from scipy.stats import mode
#from randomforest import RandomForestClassifier



class AIDataDispatcher:
    def load_model_random_forest(self, filename='random_forest_model.joblib'):
        return load(filename)

    def predict(self, X):
        # Ensure X is a NumPy array for consistent indexing
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, features_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features_indices])
        
        final_predictions, _ = mode(predictions, axis=1)
        return final_predictions.flatten()

    def predictRandomForest(self, model, df: pd.DataFrame):
        print(model)
        prediction = model.predict(df)
        print(prediction)
        return prediction

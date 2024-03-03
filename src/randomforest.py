from joblib import Parallel, delayed, dump, load
import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from src.decisiontreeclassifier import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import ParameterGrid
from shapely.geometry import Point, Polygon
import multiprocessing

class RandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        np.random.seed(self.random_state)

        indices = np.arange(n_samples)
        tree_results = Parallel(n_jobs=(multiprocessing.cpu_count() - 1), verbose=10)(
            delayed(self.train_tree)(indices, X, y, max_features, n_features, self.random_state + i if self.random_state is not None else None)
            for i in range(self.n_estimators)
        )

        self.trees.extend(tree_results)
        dump(self, 'random_forest_model.joblib')  # Save the model to a file

        print("Model saved to random_forest_model.dill")

    def predict(self, X):
        # Ensure X is a NumPy array for consistent indexing
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, features_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features_indices])
        
        final_predictions, _ = mode(predictions, axis=1)
        return final_predictions.flatten()
    
    
    def predict_with_location(self, X_df):
        points_df = X_df[X_df['Type'] == 'Point']
        # Assuming 'polygons_df' has been correctly filtered to include only polygon rows
        polygons_df = X_df[(X_df['Type'] == 'Polygon') | (X_df['group'].isin(X_df[X_df['Type'] == 'Polygon']['group'])) & (X_df['Type'] != 'Point')]

        for index, row in points_df.iterrows():
            latitude, longitude = row['Latitude'], row['Longitude']
            point = np.array([[longitude, latitude]])
            predictions = np.zeros((1, len(self.trees)))
            for i, (tree, features_indices) in enumerate(self.trees):
                predictions[:, i] = tree.predict(point[:, features_indices])
            final_prediction = mode(predictions, axis=1)[0].flatten()[0]
            X_df.at[index, 'Prediction'] = final_prediction

        grouped_polygons = polygons_df.groupby('group')
        for group_number, group_df in grouped_polygons:
            polygon_coords = [(x, y) for x, y in zip(group_df['Longitude'], group_df['Latitude'])]
            polygon = Polygon(polygon_coords)
            minx, miny, maxx, maxy = polygon.bounds
            sumOfPredictions = 0
            count = 0

            for lat in np.arange(miny, maxy, 1/69):
                for lon in np.arange(minx, maxx, 1/(np.cos(np.radians(lat)) * 69)):
                    if polygon.contains(Point(lon, lat)):
                        point = np.array([[lon, lat]])
                        predictions = np.zeros((1, len(self.trees)))
                        for j, (tree, features_indices) in enumerate(self.trees):
                            predictions[:, j] = tree.predict(point[:, features_indices])
                        sumOfPredictions += mode(predictions, axis=1)[0].flatten()[0]
                        count += 1

            # Update the prediction for this group in X_df
            if count > 0:
                average_prediction = sumOfPredictions / count
                # Find rows belonging to this group and update
                X_df.loc[X_df['group'] == group_number, 'Prediction'] = average_prediction

        return X_df

    def load_model(self, filename):
        try:
            return load(filename)
        except Exception as e:
            print("Model not found. Exception:", str(e))
            return None

    def train_tree(self, indices, X, y, max_features, n_features, random_state):
        np.random.seed(random_state)  # Ensure reproducibility for each tree
        X_subset, y_subset = X[indices], y[indices]
        features_indices = np.random.choice(n_features, size=max_features, replace=False)

        tree = DecisionTreeClassifier(min_samples_split=25, max_depth=25, feature_selection_strategy='sqrt')
        #tree = DecisionTreeClassifier(min_samples_split=1, max_depth=1, feature_selection_strategy='sqrt')
        tree.fit(X_subset[:, features_indices], y_subset)
        return tree, features_indices  # Return a tuple of the tree and its feature indices

    def evaluate_model_with_kfold(self, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = RandomForest(n_estimators=10, max_features='sqrt', random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

        return np.mean(accuracies), np.std(accuracies)




    def model_predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds

    #To make a prediction with a list of trained base learners, we will average the predicted probabilities for each class of every base learner. 
    #The average will be the predicted probability of the random forest model.
    def _predict_proba_w_base_learners(self,  X_set: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))

        return pred_prob_list

    def predict_proba(self, X_set: np.array) -> list:
        """Returns the predicted probs for a given data set"""

        pred_probs = []
        base_learners_pred_probs = RandomForest._predict_proba_w_base_learners(X_set)

        # Average the predicted probabilities of base learners
        for obs in range(X_set.shape[0]):
            base_learner_probs_for_obs = [a[obs] for a in base_learners_pred_probs]
            # Calculate the average for each index
            obs_average_pred_probs = np.mean(base_learner_probs_for_obs, axis=0)
            pred_probs.append(obs_average_pred_probs)

        return pred_probs

    def grid_search_cv(self, X, y, param_grid, n_splits=5):
        best_score = 0
        best_params = None
        for params in ParameterGrid(param_grid):
            print("Evaluating parameters:", params)
            model = DecisionTreeClassifier(
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                feature_selection_strategy=params['feature_selection_strategy']
            )
            # Use your evaluate_model_with_kfold or similar function
            mean_accuracy, _ = RandomForest.evaluate_model_with_kfold_test(X, y, model, n_splits=n_splits)
            print(f"Mean Accuracy: {mean_accuracy}")
            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_params = params
        return best_score, best_params

    def evaluate_model_with_kfold_test(self, X, y, model, n_splits=2):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
        return np.mean(accuracies), np.std(accuracies)

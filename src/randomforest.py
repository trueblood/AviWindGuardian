from joblib import Parallel, delayed, dump, load
import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
#from decisiontreeclassifier import DecisionTreeClassifier when running locally
from src.decisiontreeclassifier import DecisionTreeClassifier #when running dash app
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import ParameterGrid
from shapely.geometry import Point, Polygon
#from line_profiler import profile



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

        # Nested Function to train a single tree
        #def train_tree(seed):
        #    np.random.seed(seed)  # Ensure reproducibility for each tree
        #    indices = np.random.choice(n_samples, size=n_samples, replace=True)
        #    X_subset, y_subset = X[indices], y[indices]
        #    features_indices = np.random.choice(n_features, size=max_features, replace=False)

        #    tree = DecisionTreeClassifier(min_samples_split=2, max_depth=2, feature_selection_strategy=max_features)
        #    tree.fit(X_subset[:, features_indices], y_subset)
        #    return tree, features_indices
        
        # Parallel execution
        #tree_results = Parallel(n_jobs=15, verbose=10)(
        #    delayed(train_tree)(i+self.random_state if self.random_state is not None else None) 
        #    for i in range(self.n_estimators)
        #)
        indices = np.arange(n_samples)
        tree_results = Parallel(n_jobs=15, verbose=10)(
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
    
    '''
    def predict_with_location(self, X_df):
        #print("Predicting with location")
        #print(X_df)
        # Filter rows for Points and Polygons separately
        points_df = X_df[X_df['Type'] == 'Point']
        # Filter to include only rows related to polygons (including first row labeled 'Polygon' and subsequent rows)
        polygons_df = X_df[(X_df['Type'] == 'Polygon') | (X_df['group'].isin(X_df[X_df['Type'] == 'Polygon']['group'])) & (X_df['Type'] != 'Point')]
        #print(polygons_df)
        # Initialize an empty dictionary to hold the mapping from group numbers to coordinate lists
        # polygon_groups = {}

        print("Points:", points_df)
        print("Polygons:", polygons_df)
        
        
        #print(polygons_df)
        # Iterate over each row in the polygons dataframe
        # for index, row in polygons_df.iterrows():
        #     group_number = row['group']
        #     # Create a tuple of (longitude, latitude) for the current row
        #     coord = (row['Longitude'], row['Latitude'])

        #     # If the group number is already in the dictionary, append the coordinate to its list
        #     if group_number in polygon_groups:
        #         polygon_groups[group_number].append(coord)
        #     else:
        #         # Otherwise, initialize the list with the current coordinate
        #         polygon_groups[group_number] = [coord]
        
        # Example usage
        #for group, coords in polygon_groups.items():
        #    print(f"Group {group}: {coords}")
        
        #print("Points polygons:", polygons_df)
        # Loop through each point in points_df
        for index, row in points_df.iterrows():
            #print("in loop")
            latitude = row['Latitude']
            longitude = row['Longitude']
            
            # Prepare the point for prediction (ensure it matches your model's input shape/expectation)
            point = np.array([[longitude, latitude]])  # Assuming model expects shape (1, 2) with [longitude, latitude]
            
            # Initialize an array to hold prediction for each tree (assuming binary classification for simplicity)
            predictions = np.zeros((1, len(self.trees)))
            
            for i, (tree, features_indices) in enumerate(self.trees):
                # Direct prediction on the point; adjust if your model's prediction method differs
                predictions[:, i] = tree.predict(point[:, features_indices])
            
            # Use mode to determine the final prediction (most common prediction among all trees)
            final_prediction = mode(predictions, axis=1)[0].flatten()[0]
            #print(final_prediction)
            # Update the 'Prediction' column for the current row
            X_df.at[index, 'Prediction'] = final_prediction
            #print(X_df)
        results = []
        #print("before poly loop")

        grouped = polygons_df.groupby('group').agg(list).reset_index()
        print("grouped:", grouped)
        for i, (index, group_df) in enumerate(grouped.iterrows()):
            # Your code here
            #print("Processing group:", group_df['group'])
            #print("in loop polygon")
            #print(group_df)
            polygon_coords = [(x, y) for x, y in zip(group_df['Longitude'], group_df['Latitude'])]
            #print(polygon_coords)
            polygon = Polygon(polygon_coords)
                        
            # Get the bounding box of the polygon
            minx, miny, maxx, maxy = polygon.bounds

            # Generate points within the bounding box, approximately one mile apart
            lat_steps = np.arange(miny, maxy, 1/69)  # Approx. 1 mile steps in latitude
            long_steps = np.arange(minx, maxx, 1/(np.cos(np.radians(miny)) * 69))  # Approx. 1 mile steps in longitude, adjusted for latitude
            lat_long_steps_length = len(lat_steps)
            
            print("lat_steps:", lat_steps)
            print("long_steps:", long_steps)
            sumOfPredictions = 0
            for h in range(lat_long_steps_length):
                point = np.array([[long_steps[h], lat_steps[h]]])  # Assuming model expects shape (1, 2) with [longitude, latitude]
                predictions = np.zeros((1, len(self.trees)))
                #print("polygon contains point")
                for j, (tree, features_indices) in enumerate(self.trees):
                    #print("in tree loop")
                    # Direct prediction on the point; adjust if your model's prediction method differs
                    predictions[:, j] = tree.predict(point[:, features_indices])
                    #prediction = model_predict((lon, lat))
                    #print(predictions)
                        # Store the group number and the aggregated predictions
                    total = np.sum(predictions)   
                    sumOfPredictions += total
            results.append({
                'group_number': i,
                'predictions': sumOfPredictions
            })
            
            # for lat in lat_steps:
            #     #print("in loop lat", lat)
            #     for lon in long_steps:
            #         point = np.array([[lon, lat]])  # Assuming model expects shape (1, 2) with [longitude, latitude]
            #         predictions = np.zeros((1, len(self.trees)))
            #         #print("polygon contains point")
            #         for i, (tree, features_indices) in enumerate(self.trees):
            #             #print("in tree loop")
            #             # Direct prediction on the point; adjust if your model's prediction method differs
            #             predictions[:, i] = tree.predict(point[:, features_indices])
            #             #prediction = model_predict((lon, lat))
            #             #print(predictions)
            #              # Store the group number and the aggregated predictions
            #             results.append({
            #                 'group_number': i,
            #                 'predictions': predictions
            #             })
            
        # Print all records in results
        for record in results:
            print(record)
            
            

        # Iterate through the results to sum predictions and update original_df
        for result in results:
            group_number = result['group_number']
            prediction_total = result['predictions']  # Sum up predictions for the group
            
            print("prediction_total:", prediction_total)
            # Find the index of the first occurrence of each group in X_df
            top_row_index = X_df[X_df['group'] == group_number].index.min()  # Get the minimum index for the group
            print("top row index", top_row_index)
            # Update the 'Prediction' column for the top row of the group
            if pd.notnull(top_row_index):  # Check if the group number exists in X_df
                X_df.at[top_row_index, 'Prediction'] = prediction_total
        print(X_df)
        return X_df
'''

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

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('../datasets/dataset.csv')

    # Display the first few rows of the dataframe to understand its structure
    df.head()   

    # Define the feature names based on your dataset
    feature_names = ['xlong', 'ylat', 'type_encoded', 'birdspecies_encoded', 'timestamp']

    # Mocking up some feature importances as an example, replace with your actual feature importances
    # Assuming my_tree.feature_importances is a dictionary with indices as keys and importance scores as values
    # For demonstration, using random values
    feature_importances = {0: 0.2, 1: 0.3, 2: 0.25, 3: 0.25}

    # Since 'timestamp' might not be immediately useful and contains 'not applicable', we'll exclude it.
    # We'll also prepare to encode 'type' and 'birdspecies' if they have relevant variance.
    df = df.drop(['timestamp'], axis=1)  # Dropping the timestamp column
    df = df[df['type'] != 'bird']

    # Now, when converting this column to int, there should be no 'not applicable' values
    data = df.values  # If 'data' is derived from df
    df['birdspecies'] = df['birdspecies'].str.lower()  # Convert to lowercase for consistency
    # Check the variance of 'type' and 'birdspecies' to decide on encoding
    print(df['type'].value_counts())
    print(df['birdspecies'].value_counts())
        
    # Encode categorical variables using Label Encoding for simplicity
    label_encoder_type = LabelEncoder()
    label_encoder_birdspecies = LabelEncoder()

    
    df['type_encoded'] = label_encoder_type.fit_transform(df['type'])
    df['birdspecies_encoded'] = label_encoder_birdspecies.fit_transform(df['birdspecies'])
    df['collision'] = df['collision'].replace('not applicable', -1)

    # Example of handling 'not applicable' values in 'birdspecies_encoded' column before conversion
    df['birdspecies_encoded'] = df['birdspecies_encoded'].replace('not applicable', np.nan)  # Replace with NaN or a placeholder
    #df.dropna(subset=['birdspecies_encoded'], inplace=True)  # Drop rows with NaN in 'birdspecies_encoded'

    # Convert 'collision' to integer
    df['collision'] = df['collision'].astype(int)

    # Prepare X and Y
    #X = df[['xlong', 'ylat', 'type_encoded', 'birdspecies_encoded']].values
    X = df[['xlong', 'ylat']].values
    Y = df['collision'].values

    trainModel = False
    if trainModel:
        randomForest = RandomForest()
        mean_accuracy, std_dev = randomForest.evaluate_model_with_kfold(X, Y, n_splits=5)
        print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_dev}")
    else:
        model = RandomForest.load_model('../models/random_forest_model.joblib')
        print(model)
        # Assuming long_point and lat_point are your longitude and latitude
        test_data = pd.DataFrame({
            'xlong': [-99.787033, -99.725624, -99.769722, -99.80706, -99.758476, -99.772133, -99.740372, -99.810005, -99.762489, -99.761673, -99.776581, -99.724426, -99.714981, -99.70752, -99.760849, -99.745461, -99.733009, -99.750793, -99.814697, -99.742203, -99.737236, -99.741096, -99.781944, -99.796494, -99.733086, -99.775742, -99.724144, -99.786308, -99.737022, -99.776512, -99.741119, -99.735718, -99.762627, -99.74147, -99.804855, -99.792625, -99.788475, -99.799377, -99.79464, -99.756958, -99.790665, -99.720284, -99.714928, -99.782089, -99.821129, -99.744316, -99.731606, -99.778603, -99.782372, -99.826469, -99.751343, -99.752548, -99.775963, -99.764076, -99.812325, -99.809586, -99.752739, -99.771027, -99.720001, -99.746208, -118.364197, -118.363762, -118.36441, -93.518082, -93.632835, -93.523651, -93.623009, -93.700424, -93.430367, -92.672089, -93.51371, -93.515892, -93.367798, -70.541801, -70.545303, -70.547798, -70.545303, -93.325691, -93.428093, -93.431992, -93.354897, -93.632095, -93.636795, -83.736298, -83.736298, -94.65139, -94.707893, -94.685692, -94.683395, -94.673988, -94.695091, -94.68869, -94.61039, -94.692795, -94.706894, -94.687691, -94.659096, -94.674889, -94.68409, -94.68869, -94.665192, -94.670395, -94.618591, -94.69899, -94.723892, -94.611893, -94.658493, -94.62249, -94.669289, -94.634491, -94.666992, -94.676788, -94.677391, -94.689392, -94.71209, -94.652596, -94.613396, -94.665092, -94.703896, -94.688995, -94.726593, -94.63839, -94.655289, -94.647896, -94.612091, -94.67009, -94.619591, -94.654495, -94.683502, -94.674194, -94.614189, -94.673492, -94.630989, -94.692696, -94.670593, -94.647491, -94.615791, -94.660995, -94.715591, -94.704094, -94.696693, -94.716591, -94.722794, -94.622093, -94.695595, -94.730293, -94.618294, -94.678894, -94.676994, -94.711792, -94.606728, -94.681793, -94.635391, -94.691689, -94.659866, -94.655891, -94.655891, -94.648689, -94.630692, -94.665695, -94.626991, -94.686043, -94.697762, -94.671776, -94.649178, -94.734169, -94.724846, -94.72422, -94.662544, -94.664101, -94.664093, -94.647606, -94.725616, -94.639961, -94.678551, -94.665192, -94.716476, -94.669571, -94.659653, -94.712288, -94.705017, -94.673912, -94.713112, -94.664482, -94.728653, -94.695923, -94.66095, -94.657524, -94.712463, -94.649925, -94.667892, -94.634834, -94.728088, -94.697464, -94.616638, -94.721962, -94.65696, -94.615784, -94.716339],
            'ylat': [36.501724, 36.437126, 36.444931, 36.513935, 36.444984, 36.431931, 36.489838, 36.476582, 36.454903, 36.502792, 36.485386, 36.491375, 36.490211, 36.490849, 36.429668, 36.448009, 36.489468, 36.488712, 36.48975, 36.432888, 36.498882, 36.423683, 36.433651, 36.503357, 36.451591, 36.445465, 36.421703, 36.44593, 36.435909, 36.429882, 36.50259, 36.423359, 36.515823, 36.451477, 36.476624, 36.474033, 36.476807, 36.440704, 36.438831, 36.491249, 36.44141, 36.424934, 36.424843, 36.473541, 36.517494, 36.513348, 36.44109, 36.502522, 36.484463, 36.515972, 36.446674, 36.428902, 36.458096, 36.440956, 36.513859, 36.495914, 36.504105, 36.456665, 36.489346, 36.427345, 35.077644, 35.077908, 35.077435, 42.01363, 41.882477, 42.006813, 41.88147, 41.977608, 42.028233, 41.742046, 42.019119, 42.016373, 42.49794, 41.752491, 41.754192, 41.75959, 41.757591, 42.20639, 42.146091, 42.145592, 41.904194, 42.335491, 42.335491, 41.382492, 41.384693, 41.421494, 41.470394, 41.471092, 41.484993, 41.494892, 41.457893, 41.486591, 41.441395, 41.499294, 41.440792, 41.445091, 41.466694, 41.423492, 41.443592, 41.499294, 41.452293, 41.434994, 41.466793, 41.457794, 41.463192, 41.452892, 41.433193, 41.466595, 41.452293, 41.439693, 41.482494, 41.452393, 41.463093, 41.470192, 41.468391, 41.455894, 41.467094, 41.423695, 41.499695, 41.435894, 41.442894, 41.439693, 41.466293, 41.465393, 41.478592, 41.495193, 41.451992, 41.432995, 41.433804, 41.434193, 41.441395, 41.463093, 41.481194, 41.434994, 41.423992, 41.443691, 41.452694, 41.421093, 41.441792, 41.471092, 41.499092, 41.469791, 41.443394, 41.441994, 41.445194, 41.442394, 41.441593, 41.443993, 41.473293, 41.441292, 41.441357, 41.471592, 41.449795, 41.445091, 41.457573, 41.421394, 41.457294, 41.455395, 41.439194, 41.470993, 41.481194, 40.907246, 40.927402, 40.91991, 40.947777, 40.903179, 40.906811, 40.91663, 40.91256, 40.952431, 40.931583, 40.912476, 40.925251, 40.918289, 40.951794, 40.902493, 40.913757, 40.902508, 40.925396, 40.912891, 40.916325, 40.955795, 40.929382, 40.9258, 40.905937, 40.901394, 40.906422, 40.913597, 40.923447, 40.917538, 40.919884, 40.90395, 40.916672, 40.908005, 40.913837, 40.927204, 40.905529, 40.906044, 40.92387]
        })
        print("Length of xlong:", len(test_data['xlong']))
        print("Length of ylat:", len(test_data['ylat']))
        # Make a prediction with the model
        prediction = model.predict(test_data)
        print(prediction)
    

'''
    param_grid = {
        'max_depth': [5, 10],
        'min_samples_split': [1, 2],
        'feature_selection_strategy': ['sqrt', 'log2']
    }

    # Assuming X, Y are your features and labels
    best_score, best_params = grid_search_cv(X, Y, param_grid)
    print("Best Score:", best_score)
    print("Best Parameters:", best_params)
'''


    #model = load('random_forest_model.joblib')
    #predictions = model.predict(X_test)
    #else:
     #   model = load_model('random_forest_model.joblib')
        #predictions = model.predict(X_test)

    
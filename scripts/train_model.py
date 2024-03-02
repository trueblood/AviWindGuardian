import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.randomforest import RandomForest
import numpy as np

def trainRandomForestModel(): 
    df = pd.read_csv('./src/datasets/turbines/wind_turbine_collisions.csv')

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
    randomForest = RandomForest()

    mean_accuracy, std_dev = randomForest.evaluate_model_with_kfold(X, Y, n_splits=5)
    print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_dev}")
    
def main():
    trainRandomForestModel()

if __name__ == "__main__":
    main()


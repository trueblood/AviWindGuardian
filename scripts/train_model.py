import pandas as pd
from src.randomforest import RandomForest
import numpy as np

def trainRandomForestModel(): 
    df = pd.read_csv('./src/datasets/turbines/wind_turbines_with_collisions.csv')

    # Display the first few rows of the dataframe to understand its structure
    df.head() 
    
    # Replace 'not applicable' with -1
    df['collision'] = df['collision'].replace('not applicable', -1)

    # Fill NaN values with -1 (or an appropriate value for your context)
    df['collision'] = df['collision'].fillna(-1)
    
    # Convert 'collision' to integer
    df['collision'] = df['collision'].astype(int)

    # Prepare X and Y
    X = df[['xlong', 'ylat']].values
    Y = df['collision'].values
    
    randomForest = RandomForest()
    mean_accuracy, std_dev = randomForest.evaluate_model_with_kfold(X, Y, n_splits=5)
    
    print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_dev}")
    
def main():
    trainRandomForestModel()

if __name__ == "__main__":
    main()


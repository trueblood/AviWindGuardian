import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

class CollisionForecastARIMA:
    def __init__(self, data, columns):
        """
        Initializes the CollisionForecastARIMA object with collision data and column names.
        
        Parameters:
        - data: DataFrame containing collision data.
        - columns: A list of column names to be used in the ARIMA model.
        """
        self.data = data
        self.columns = columns
        self.model_fit = None

    def prepare_data(self):
        """
        Prepares the time series data by aggregating collision counts per day based on specified columns.
        """
        # Assumes the first column in self.columns is the time column for aggregation
        time_column = self.columns[0]
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        self.collision_counts = self.data.groupby(self.data[time_column].dt.date).size()
        self.collision_counts.index = pd.to_datetime(self.collision_counts.index)
    
    def fit_model(self, order=(1,1,1)):
        """
        Fits an ARIMA model to the prepared time series data.
        
        Parameters:
        - order: A tuple (p, d, q) representing the ARIMA model parameters.
        """
        self.model = ARIMA(self.collision_counts, order=order)
        self.model_fit = self.model.fit()

    def forecast(self, steps=10):
        """
        Forecasts future collision counts.
        
        Parameters:
        - steps: The number of future time periods to forecast.
        
        Returns:
        - A forecast object containing the forecasted values.
        """
        if self.model_fit is None:
            print("Model is not fitted. Call fit_model() first.")
            return None
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def load_model(FileName):
        """
        Loads a pickled ARIMA model from a file.

        Parameters:
        - FileName: The filename of the pickled ARIMA model.

        Returns:
        - A model object containing the loaded ARIMA model.
        """
        with open(FileName, 'rb') as model_file:
            model = pickle.load(model_file)
        return model

# Usage example
if __name__ == "__main__":
    # Load your data here
    data = pd.read_csv('path_to_your_collision_data.csv')
    
    # Specify the column names (time column first)
    columns = ['Collision_Time']
    
    # Initialize the forecasting object with your data and column names
    forecast_arima = CollisionForecastARIMA(data, columns)
    
    # Prepare the data
    forecast_arima.prepare_data()
    
    # Fit the ARIMA model
    forecast_arima.fit_model(order=(1, 1, 1))

    # Save the model to a file
    model_filename = 'arima_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model_fit, model_file)
    
    # Forecast future collision counts
    future_collisions = forecast_arima.forecast(steps=10)
    print(future_collisions)

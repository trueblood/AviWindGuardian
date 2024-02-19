import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class CollisionForecast:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_data()
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        self.collision_data = df['Collision Count']

    def fit_model(self, order=(2,0,1)):
        self.model = ARIMA(self.collision_data, order=order)
        self.model_fit = self.model.fit()

    def forecast_collisions(self, steps=30):
        if not self.model_fit:
            print("Model is not fitted yet. Please call fit_model() first.")
            return
        self.forecast = self.model_fit.forecast(steps=steps)
        return self.forecast

    def adjust_forecast(self, increase_factor=1.0, decrease_factor=1.0):
        self.forecast_increased = self.forecast * increase_factor
        self.forecast_decreased = self.forecast * decrease_factor

    def plot_forecast(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.collision_data.index, self.collision_data, label='Actual Collisions', color='blue')
        plt.plot(self.forecast.index, self.forecast, label='Forecast', color='green')
        plt.plot(self.forecast.index, self.forecast_increased, label='Increased Turbines Forecast', color='red', linestyle='--')
        plt.plot(self.forecast.index, self.forecast_decreased, label='Decreased Turbines Forecast', color='purple', linestyle='--')
        plt.title('Collision Count Forecast with Adjusted Turbine Numbers')
        plt.xlabel('Date')
        plt.ylabel('Collision Count')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    forecast = CollisionForecast('collision_data.csv')
    forecast.fit_model()
    forecast.forecast_collisions()
    forecast.adjust_forecast(increase_factor=1.10, decrease_factor=0.90)  # 10% increase, 10% decrease
    forecast.plot_forecast()
         
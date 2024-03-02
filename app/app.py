# -*- coding: utf-8 -*-
import json
import random
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import dash_leaflet as dl
from dash import Dash, html, Output, Input
from dash.exceptions import PreventUpdate
from dash_extensions.javascript import assign
import dash_leaflet as dl
from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc
import pandas as pd
#from scripts.ai_data_dispatcher import AIDataDispatcher
from src.randomforest import RandomForest
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import warnings
from datetime import datetime, timedelta
from prophet import Prophet
from dash.dependencies import Input, Output, State, MATCH, ALL
import requests
import re

warnings.filterwarnings("ignore")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP],
)

# Load the detailed collision data
detailed_collisions_path = '../src/datasets/turbines/detailed_wind_turbine_collisions.csv'
detailed_collisions_df = pd.read_csv(detailed_collisions_path)

# Ensure 'Timestamp' is a datetime type and extract the year
detailed_collisions_df['Timestamp'] = pd.to_datetime(detailed_collisions_df['Timestamp'], errors='coerce')
detailed_collisions_df['Year'] = detailed_collisions_df['Timestamp'].dt.year

# Determine the minimum and maximum year in the dataset
min_year = int(detailed_collisions_df['Year'].min())
max_year = int(detailed_collisions_df['Year'].max() - 1)

"""
==========================================================================
Markdown Text
"""
learn_text = dcc.Markdown(
    """
- **Learn**: Click on the learn tab to learn about the product.

- **Point Marker**: Click on the point marker tab to plot points.

    - **Map Interaction**: Select the map and click on either a mark point or plot a polygon.
        - To erase all points, click the trash can icon.
        - To move and modify existing points and polygons, click the edit button.

    - **Wind Speed and Collision Percentage**:
        - After your point is added to the map, the code retrieves the wind speed from a weather API service and sends the longitude and latitude coordinates to the model, which returns the collision percentage.

            > **Note**: The polygons take longer to process because the code considers the longitudes and latitudes inside the polygon.

    - **Model Feedback**:
        - After results come back, you have the option to provide feedback to the model.
        - Press thumbs up if the collision prediction is correct; otherwise, press thumbs down. This adds or updates the longitude and latitude coordinates in the dataset that trains the turbines.

- **Map Data**:
    - Hover over each point on the map to get average wind speed and collision risk.

- **Turbine Location**:
    - After you add a new turbine location or plot a turbine location area, the forecast collision line chart updates with the new points, adding Time Series Analysis to the data.
        - You can filter the year and the forecasting period.
        - Add more and remove points to see how that affects your collision prediction.

- **Model Retraining**:
    - After collecting enough model feedback data, you can go in and retrain the random forest model.
        - Note: It takes a while to retrain the models.
    - The forecast model requires a different training set. You will need to run Python scripts behind the scenes to set this up.
        - An option to retrain this model from the dashboard is included in case you run into any errors with forecasting.

> **Note**: If you encounter any errors with the models, go to the model updates tab and retrain the models. Please note this takes a while.

    """
)

learn_text_model_training = dcc.Markdown(
    """
     This page lets you train both the Random Forest and Forecasting models. 
     If there is new turbine collision data to train on, you want to click the 'Train Random Forest Model' button. 
     Else, if you want to train the Forecasting model on known bird collisions with date for turbines on updated data, click the 'Train Forecasting Model' button. 
     Or, if you're having troubleshooting issues, retraining the models is usually a fix. Warning this may take. 
    """
)

footer = html.Div(
    dcc.Markdown(
        """
        AviWind Guardian provides this content for informational purposes only, aiming to balance renewable energy and avian conservation through machine learning, 
        without guaranteeing the information's accuracy or serving as a substitute for professional advice, and disclaims liability for decisions made based on its use.
        """,
        style={'font-size': '8pt'}
    ),
    className="p-2 mt-5 bg-primary text-white small",
)

"""
==========================================================================
Map
"""

def make_map():
    """
    Function to create a map with specific edit controls.
    """
    return dl.Map(center=[39.5501, -105.7821], zoom=6, children=[
        dl.TileLayer(), 
        dl.LayerGroup(id="marker-layer"), 
        dl.GeoJSON(id="geojson"),
        dl.EasyButton(icon="fa-trash", title="Clear All Marker Points", id="btn"),
        dl.FeatureGroup([
            dl.EditControl(
                id="edit_control", 
                position="topleft",
                draw={
                    "polyline": False,  # Disable line drawing
                    "polygon": True,    # Keep polygon drawing
                    "circle": False,    # Disable circle drawing
                    "rectangle": False, # Disable rectangle drawing
                    "marker": True,     # Enable marker drawing
                    "circlemarker": False # Disable circlemarker drawing
                }
            )
        ], id="feature-group")
    ], style={'width': '100%', 'height': '50vh', "display": "inline-block"}, id="map")


"""
==========================================================================
Figures
"""

# shows all birds combined
# def load_forecast(_):
#     print("in load forecast")
#     # Assuming the ARIMA model and necessary libraries are already imported

#     # Load the fitted model
#     try:
#         data_path = 'datasets/turbines/detailed_wind_turbine_collisions_bk.csv'
#         data = pd.read_csv(data_path)

#         # Ensure 'Timestamp' is in datetime format
#         data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
#         columns = ['Timestamp']
        
#         # Load the saved model
#         model_filename = 'arima_model_forecasting.joblib'
#         forecast_arima = CollisionForecastARIMA(data, columns)
        
#         model = forecast_arima.load_model(model_filename)
        
#         # Forecast future collision counts
#         steps = 15
#         future_collisions = model.forecast(steps=steps)
        
#         # Generate a date range for the forecasted data
#         last_date = data['Timestamp'].max()
#         forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
    
#         # Assign this new datetime index to your forecasted data
#         future_collisions.index = forecast_dates  # Align forecast with historical data's timeline

#         # Create a figure to plot the data
#         fig = go.Figure()

#         # Aggregate collision counts per day
#         data['Date'] = data['Timestamp'].dt.date
#         aggregate_counts = data.groupby('Date').size()

#         # Plot historical aggregated collision counts
#         fig.add_trace(go.Scatter(x=aggregate_counts.index, y=aggregate_counts, mode='lines', name='Historical Aggregated Collisions'))

#         # Add the forecasted data
#         fig.add_trace(go.Scatter(x=forecast_dates, y=future_collisions, mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))

#         # Update plot layout
#         fig.update_layout(
#             title='Aggregated Collision Counts and Forecast',
#             xaxis_title='Date',
#             yaxis_title='Number of Collisions',
#             xaxis_rangeslider_visible=True,
#             showlegend=True,
#             template="none"
#         )

#         return fig, 'Model loaded and forecast generated successfully.'
#     except FileNotFoundError:
#         return go.Figure(), 'Model file not found. Please fit the model first.'

# shows all birds aggreated 
# def load_forecast(_):
#     print("in load forecast")
#     # Assuming the ARIMA model and necessary libraries are already imported

#     # Load the fitted model
#     try:
#         data_path = 'datasets/turbines/detailed_wind_turbine_collisions_bk.csv'
#         data = pd.read_csv(data_path)

#         # Ensure 'Timestamp' is in datetime format
#         data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
#         columns = ['Timestamp']
#         # Load the saved model (example path, adjust as necessary)
#         model_filename = 'arima_model_forecasting.joblib'
#         forecast_arima = CollisionForecastARIMA(data, columns)
#         # Example: model = load_your_model_function(model_filename)
#         model = forecast_arima.load_model(model_filename)
#         # Forecast future collision counts (adjust 'steps' as necessary)
#         future_collisions = model.forecast(steps=10)
#         last_date = data['Timestamp'].max()
#         forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')
    
#         # Now, assign this new datetime index to your forecasted data
#         future_collisions.index = forecast_dates  # Align forecast with historical data's timeline

#         # Create a figure to plot the data
#         fig = go.Figure()

#         # Plot historical data for each bird species
#         for species in data['Species'].unique():
#             species_data = data[data['Species'] == species]
#             species_counts = species_data.groupby(species_data['Timestamp'].dt.date).size()
#             fig.add_trace(go.Scatter(x=species_counts.index, y=species_counts, mode='lines', name=f'{species} - Historical'))

#         # Assuming 'future_collisions' includes a datetime index matching your historical data
#         # Add the forecasted data
#         forecast_index = future_collisions.index  # Adjust as necessary
#         fig.add_trace(go.Scatter(x=forecast_index, y=future_collisions, mode='lines+markers', name='Aggregated Forecast', line=dict(dash='dot')))

#         # Update plot layout
#         fig.update_layout(
#             title='Collision Counts and Forecast by Bird Species',
#             xaxis_title='Date',
#             yaxis_title='Number of Collisions',
#             xaxis_rangeslider_visible=True,
#             showlegend=True,
#             template="none"
#         )

#         return fig, 'Model loaded and forecast generated successfully.'
#     except FileNotFoundError:
#         return go.Figure(), 'Model file not found. Please fit the model first.'

def parse_custom_date(date_str):
    #print(date_str)
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        #print(f"Failed to parse {date_str}: {e}")
        return None
# Meta Prophet aggregated 
# def load_forecast(_):
#     #print("in load forecast")
#     try:
#         # Load the detailed collision data
#         detailed_collisions_path = 'exported_data.csv'
#         detailed_collisions_df = pd.read_csv(detailed_collisions_path)
#         detailed_collisions_df['Turbine_Longitude'] = detailed_collisions_df['Turbine_Longitude'].round(4)
#         detailed_collisions_df['Turbine_Latitude'] = detailed_collisions_df['Turbine_Latitude'].round(4)
#         # Load the wind turbine location data
#         wind_turbines_path = 'datasets/turbines/wind_turbines_with_collisions.csv'
#         wind_turbines_df = pd.read_csv(wind_turbines_path)
#         wind_turbines_df['xlong'] = wind_turbines_df['xlong'].round(4)
#         wind_turbines_df['ylat'] = wind_turbines_df['ylat'].round(4)
#         # Merge the datasets on longitude and latitude
#         merged_df = pd.merge(detailed_collisions_df, wind_turbines_df, 
#                      left_on=['Turbine_Longitude', 'Turbine_Latitude'], 
#                      right_on=['xlong', 'ylat'], 
#                      how='inner')
        
#         merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], errors='coerce')
#         # Assuming 'Timestamp' column exists and represents when collisions occurred
#         #print(merged_df.head()) 
        
#         print(len(merged_df))

#         # Aggregate collision counts by date for the merged dataset
#         aggregated_data = merged_df.resample('D', on='Timestamp').agg({'collision': 'sum'}).reset_index()
#         aggregated_data.rename(columns={'Timestamp': 'ds', 'collision': 'y'}, inplace=True)
#         #print(aggregated_data.head())
#         aggregated_data = aggregated_data.dropna()

#         # Initialize and fit the Prophet model
#         model = Prophet()
#         model.fit(aggregated_data)
#         #print("after prophet fit")
        
#         # Create future dataframe for forecasting (e.g., next 365 days)
#         future_dates = model.make_future_dataframe(periods=365)
        
#         # Predict the values for future dates
#         forecast = model.predict(future_dates)
        
#         # Create a figure to plot the forecast
#         fig = go.Figure()
        
#         # Plot the historical aggregated collision counts
#         fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Aggregated Collisions'))
        
#         # Add the forecasted data
#         fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))
        
#         # Update plot layout
#         fig.update_layout(
#             title='Aggregated Collision Counts and Forecast',
#             xaxis_title='Date',
#             yaxis_title='Number of Collisions',
#             xaxis_rangeslider_visible=True,
#             showlegend=True,
#             template="none"
#         )

#         return fig, 'Model loaded and forecast generated successfully.'
#     except FileNotFoundError as e:
#         print(e)
#         return go.Figure(), 'Required file not found. Please check the file paths.'

def load_forecast(_):
    try:
        # Load the detailed collision data
        #detailed_collisions_path = 'exported_data.csv'
        #detailed_collisions_df = pd.read_csv(detailed_collisions_path)
        
        # Ensure 'Timestamp' is a datetime type and extract the year
        detailed_collisions_df['Timestamp'] = pd.to_datetime(detailed_collisions_df['Timestamp'], errors='coerce')
        detailed_collisions_df['Year'] = detailed_collisions_df['Timestamp'].dt.year

        print(len(detailed_collisions_df))
        # Assuming each row represents a collision, count collisions by year
        aggregated_data = detailed_collisions_df.groupby('Year').size().reset_index(name='y')
        aggregated_data.rename(columns={'Year': 'ds'}, inplace=True)
        aggregated_data = aggregated_data.dropna()
        aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'].astype(int).astype(str) + '-01-01')

        print(len(aggregated_data))
        print(aggregated_data.head())
        # Initialize and fit the Prophet model
        model = Prophet(yearly_seasonality=True)  # Enable yearly seasonality as we're dealing with yearly data
        model.fit(aggregated_data)

        # Create future dataframe for forecasting next years
        future_years = model.make_future_dataframe(periods=5, freq='Y')  # Forecasting for the next 5 years as an example

        # Predict the values for future years
        forecast = model.predict(future_years)
        
        # Create a figure to plot the forecast
        fig = go.Figure()
        
        # Plot the historical aggregated collision counts
        fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Collisions'))

        # Add the forecasted data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))

        # Update plot layout
        fig.update_layout(
            title='Yearly Collision Counts and Forecast',
            xaxis_title='Year',
            yaxis_title='Number of Collisions',
            xaxis_rangeslider_visible=True,
            showlegend=True,
            template="none",
            xaxis=dict(
                range=[min_year, max_year],  # replace 'start_year' with the actual start year
            ),
        )

        return fig, 'Model loaded and forecast generated successfully.'
    except FileNotFoundError as e:
        print(e)
        return go.Figure(), 'Required file not found. Please check the file paths.'
    
# not working
# def load_forecast(_):
#     print("in load forecast")

#     try:
#         data_path = 'datasets/turbines/detailed_wind_turbine_collisions_bk_mod.csv'
#         data = pd.read_csv(data_path)
#         #data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
#         #data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y', errors='coerce')
#         #data['Timestamp'] = data['Timestamp'].dt.date
#         #data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce', infer_datetime_format=True)
#         #data['Timestamp'] = data['Timestamp'].dt.date
#         #data['Timestamp'] = data['Timestamp'].apply(lambda x: parse_custom_date(x))    
#         # Ensure 'Timestamp' is in datetime format
#         #data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce', infer_datetime_format=True)

#         # Convert datetime objects to the desired string format 'YYYY-MM-DD HH:MM:SS'
#        # data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

#         # Display the first few rows to verify the changes
#         #data.head()
        
        
        
#         #data.to_csv('exported_data.csv', index=False)

#         print("after lamda")
#         none_count = data['Timestamp'].isnull().sum()
#         print(f"Number of None values after custom parsing: {none_count}")
#         data['Timestamp'] = data['Timestamp'].replace({None: np.nan})
#         data.to_csv('exported_data.csv', index=False)
        
        
#         columns = ['Timestamp']
#         print(data['Timestamp'].dtype)
#         overall_max_timestamp = data['Timestamp'].max()
#         print(overall_max_timestamp)
#         print("data loaded", data.head())
        
        
                
#         model_filename = 'arima_model_forecasting.joblib'
        
#         # Initialize a figure for plotting
#         fig = go.Figure()

#         # Iterate over each bird species to forecast individually
#         for species in data['Species'].unique():
#             species_data = data[data['Species'] == species]
#             # Debugging print: Check if species data is empty
#             if species_data.empty:
#                 print(f"No data found for species: {species}")
#                 continue

#             nat_rows = species_data[species_data['Timestamp'].isna()]

#             if not nat_rows.empty:
#                 print(f"Found {len(nat_rows)} rows with NaT 'Timestamp' for 'Setophaga striata'.")

#             # If necessary, inspect a few rows to understand the issue
#             if len(nat_rows) > 0:
#                 print(nat_rows.head())

#             # Further debugging to confirm 'Timestamp' conversion
#             print(f"Data available for {species}: {len(species_data)} records")
#             last_date = species_data['Timestamp'].max()
#             print(f"Last date for {species}: {last_date}")  # Debugging print
            
#             if not species_data.empty:
#                 print(f"Species data for {species} is not empty.")
#                 species_max_timestamp = species_data['Timestamp'].max()
#                 print(species_max_timestamp)
#                 species_data['Timestamp'] = pd.to_datetime(species_data['Timestamp'], errors='coerce')
#                 print("before overall_max_timestamp")
#                 #overall_max_timestamp = data['Timestamp'].max()
#                 print("after overall_max_timestamp")
                
                
                
                                
                
#                 # Convert the strings to datetime objects
#                 datetime1 = datetime.strptime(species_max_timestamp, '%Y/%m/%d %H:%M:%S')
#                 datetime2 = datetime.strptime(overall_max_timestamp, '%Y/%m/%d %H:%M:%S')

#                 # Get the date part of each datetime
#                 date1 = datetime1.date()
#                 print(date1)
#                 date2 = datetime2.date()
#                 print(date2)

#                 # Subtract the two dates
#                 date_diff = date2 - date1

#                 print(date_diff)
#                 days_to_forecast = date_diff.days + 10 
#                 print("days to forecast", days_to_forecast)
                
                
                
                
                
#                 #species_max_timestamp = pd.to_datetime(species_max_timestamp)
#                 #print(overall_max_timestamp)
#                 #print(species_max_timestamp)
#                 #time_difference = overall_max_timestamp - species_max_timestamp
#                 #print("time difference", time_difference)
#                 #days_to_forecast = time_difference.days + 10 
#                 #print(days_to_forecast)

                
                
                
                
                
                
                
                
                
                
                
                
                
#                 # Initialize the forecasting object for this species
#                 #forecast_arima = CollisionForecastARIMA(species_data, columns)
#                 forecast_arima = CollisionForecastARIMA(species_data, ['Timestamp'])

#                 # Load the model for this species
#                 model = forecast_arima.load_model(model_filename)
#                 print("model loaded")
#                 # Perform forecasting for this species
#                # Perform forecasting for this species
#                 future_collisions = model.forecast(steps=days_to_forecast)
#                 print("after future_collisions", future_collisions)

#                 # Manually create a list of forecast dates
#                 forecast_dates_list = [species_max_timestamp + pd.Timedelta(days=i+1) for i in range(days_to_forecast)]

#                 # Convert the list to a pandas DatetimeIndex
#                 forecast_dates_index = pd.DatetimeIndex(forecast_dates_list)
#                 print("after forecast_dates", forecast_dates_index)

#                 # Assign the forecast dates index to the future_collisions Series or DataFrame
#                 future_collisions.index = forecast_dates_index
#                 print("future_collisions indexed", future_collisions)

#                 # Group historical data by date and count occurrences for each species
#                 species_counts = species_data.groupby(species_data['Timestamp'].dt.date).size()
#                 print(species_counts)

#                 # Plot historical data for this species
#                 fig.add_trace(go.Scatter(x=species_counts.index, y=species_counts, mode='lines', name=f'{species} - Historical'))

#                 # Plot forecasted data using the manually created forecast dates
#                 fig.add_trace(go.Scatter(x=forecast_dates_index, y=future_collisions, mode='lines+markers', name=f'{species} - Forecast', line=dict(dash='dot')))

#         # Update plot layout
#         fig.update_layout(
#             title='Collision Counts and Forecast by Bird Species',
#             xaxis_title='Date',
#             yaxis_title='Number of Collisions',
#             xaxis_rangeslider_visible=True,
#             showlegend=True,
#             template="none"
#         )

#         return fig, 'Model loaded and forecast generated successfully for each species.'
#     except FileNotFoundError:
#         return go.Figure(), 'Model file not found. Please fit the model first.'


# not used, original 
# def load_forecast(_):
#     print("in load forecast")
#     # Load the fitted model
#     try:
#         data_path = 'datasets/turbines//detailed_wind_turbine_collisions_bk.csv'
    
#         data = pd.read_csv(data_path)
#         print("data loaded", data.head())
        
#         # Simulating the loading of data with a 'BirdSpecies' column for the example
#         #data = pd.DataFrame({
#         ##    'Timestamp': pd.date_range(start='2023-01-01', periods=120, freq='D'),
#         #    'Collisions': [i + (i % 10) for i in range(120)],
#         #    'BirdSpecies': ['Hawk' if i % 2 == 0 else 'Sparrow' for i in range(120)]
#         #})
        
#         # Specify the column names (time column first)
#         columns = ['Timestamp']
        
#         # Initialize the forecasting object with your data and column names
#         forecast_arima = CollisionForecastARIMA(data, columns)
        
#         # Prepare the data
#         #forecast_arima.prepare_data()
        
#         # Fit the ARIMA model
#         #forecast_arima.fit_model(order=(1, 1, 1))
        
#         model_filename = 'arima_model_forecasting.joblib'
#         #forecast_arima.save_model(model_filename)

#         # Save the model to a file
#         print("before model load")
#         model = forecast_arima.load_model(model_filename)
        
#         # Forecast future collision counts
#         future_collisions = model.forecast(steps=10)
#         print(future_collisions)

#     # Plotting historical data and forecasted values
#         fig = go.Figure()
#         # Plot historical data for each bird species
#         for species in data['Species'].unique():
#             species_data = data[data['Species'] == species]
#             # Group by date and count collisions
#             species_data['Timestamp'] = pd.to_datetime(species_data['Timestamp'], errors='coerce')
#             species_counts = species_data.groupby(species_data['Timestamp'].dt.date).size()
#             fig.add_trace(go.Scatter(x=species_counts.index, y=species_counts, mode='lines', name=f'{species} - Historical'))

#         # Add the aggregated forecast data
#         # Assuming future_collisions is a Series with datetime index and collision counts
#         forecast_index = future_collisions.index
#         fig.add_trace(go.Scatter(x=forecast_index, y=future_collisions, mode='lines+markers', name='Aggregated Forecast', line=dict(dash='dot')))

#         # Update layout
#         fig.update_layout(title='Collision Counts and Forecast by Bird Species',
#                         xaxis_title='Date',
#                         yaxis_title='Number of Collisions',
#                         xaxis_rangeslider_visible=True,
#                         showlegend=True,
#                         template="none")
        
        


#         return fig, 'Model loaded and forecast generat  ed successfully.'
#     except FileNotFoundError:
#         return go.Figure(), 'Model file not found. Please fit the model first.'



"""
==========================================================================
Make Tabs
"""

# =======Play tab components

cords_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Div(id="coords-display-container", style={'overflowY': 'scroll', 'height': '50vh'}),
                html.Div(id="model-feedback-container")
            ]
        )
    ],
    className="mt-4",
)

def make_coords_table(coords):
    """Generate an HTML table based on coordinates."""
    df_table = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
    return dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)


slider_card_forecast = dbc.Card(
    [
        html.H4("Adjust Forecast Parameters:", className="card-title"),
        html.Div([
            dbc.Card(
                [
                    html.H5("Forecast Period (Days):", className="card-title"),
                    dcc.Slider(
                        id="forecast-period-slider",
                        marks={i: f"{i} days" for i in range(0, 366, 30)},  # Adjust based on your needs
                        min=0,
                        max=365,
                        step=1,
                        value=0,  # Default to 0 days
                        included=False,
                    ),
                ],
                body=True,
                className="mt-2",
            ),
            dbc.Card(
                [
                    html.H5("Select Start Year:", className="card-title"),
                    dcc.Slider(
                        id="start-year-slider",
                        marks={i: str(i) for i in range(min_year, max_year + 1)},   # Example range
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=min_year,  # Default to 2021
                        included=False,
                    ),
                ],
                body=True,
                className="mt-4",
            ),
        ]),
    ],
    body=True,
    className="mt-4",
)

# ===== Model Train Tab Components

model_training_card = dbc.Card(
    [
        dbc.CardHeader("Model Training"),
        dbc.CardBody([
            # Instructional Div
            html.Div([
               html.P(learn_text_model_training)
            ], className="mb-4"),  # Adding some bottom margin for spacing
            #html.Br(),
            # Training buttons with Block style
            # html.Div(
            #     [   
            #         dbc.Button("Train Random Forest Model", id='train-button-random-forest', n_clicks=0, color="primary", className="me-1"),
            #         dbc.Button("Train Forecasting Model", id='train-button-forecast', n_clicks=0, color="primary", className="me-1"),
            #     ],
            #     className="d-grid gap-2",
            # ),
            html.Br(),
            # Output and Status Divs
            html.Div(id='model-status'),
            # Training Status Table
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Machine Learning Models"), html.Th("Status")]))] +
                [html.Tbody([html.Tr([html.Td(dbc.Button("Train Random Forest Model", id='train-button-random-forest', n_clicks=0, color="primary", className="me-1, btn-table-width")), html.Td(html.Div(id='train-status', children="Not Training"))]),
                             html.Tr([html.Td(dbc.Button("Train Forecasting Model", id='train-button-forecast', n_clicks=0, color="primary", className="me-1, btn-table-width")), html.Td(html.Div(id='output-container-button', children="Not Training"))])])],
                bordered=True,  # Add borders to the table for clarity
                hover=True,  # Enable hover effect for table rows
                responsive=True,  # Make the table responsive
                striped=False,  # Zebra-striping for table rows
            ),
        ])
    ],
    className="mt-4",
)

# ========= Learn Tab  Components
learn_card = dbc.Card(
    [
        dbc.CardHeader("An Introduction to AviWind Guardian"),
        dbc.CardBody(learn_text, style={'overflow': 'scroll', 'height': '100vh'}),
    ],
    className="mt-4",
)


# ========= Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(learn_card, tab_id="tab1", label="Learn"),
        dbc.Tab(
            [cords_card, slider_card_forecast],
            tab_id="tab-2",
            label="Plot Points",
            className="pb-4",
        ),
        dbc.Tab([model_training_card], tab_id="tab-3", label="Model Update")
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)

"""
===========================================================================
Main Layout
"""

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "AviWind Guardian",
                    className="text-center bg-primary text-white p-2",
                ),
            ),
        ),
        dbc.Row(
                dbc.Col(
                   dcc.Loading(
                    id="loading-icon",
                    children=[html.Div(id="loading-output")],
                    type="default", # This determines the style of the loading spinner
                    )
                   )
                ),
        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
                        html.Div([
                            make_map()  # Call the function to create the map
                        ]),
                        html.Hr(),
                        dcc.Graph(id='forecast-graph', className="pb-4", figure=load_forecast(1)[0],
                                config={
                                    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'zoom2d', 'autoScale2d', 'resetScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
                                    'modeBarButtonsToAdd': ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'toImage']
                                }),
                        html.Div(id="prediction-output"),
                        html.Div(id='coords-json', style={'display': 'none'}),
                        dcc.Interval(
                            id='init-trigger',
                            interval=1,  # in milliseconds
                            n_intervals=0,
                            max_intervals=1  # Stop after the first call
                        ),
                    ],
                    width=12,
                    lg=7,
                    className="pt-4",
                ),
            ],
            className="ms-1",
        ),
        dbc.Row(dbc.Col(footer)),
    ],
    fluid=True,
)


"""
==========================================================================
Callbacks
"""
# # Trigger mode (draw marker).
# @app.callback(Output("edit_contmaxBounds=[[-15.0, -170.0], [70.0, -50.0]]ol", "drawToolbar"), Input("draw_marker", "n_clicks"))
# def trigger_mode(n_clicks):
#     if n_clicks is None:
#         raise PreventUpdate
#     return dict(mode="marker", n_clicks=n_clicks)

# # Trigger mode (edit) + action (remove all).
# @app.callback(
#     [
#         Output("edit_control", "editToolbar"),  # For edit control toolbar update
#         Output("coords-display-container", "children", allow_duplicate=True),  # To clear display container
#         Output("coords-json", "children", allow_duplicate=True)  # To clear JSON data
#     ],
#     [Input("clear_all", "n_clicks")],
#     prevent_initial_call=True
# )
# def trigger_action(n_clicks):
#     if n_clicks is None:
#         raise PreventUpdate
#     # Return update for edit control toolbar, empty children for coords-display-container, and empty JSON
#     return dict(mode="remove", action="clear all", n_clicks=n_clicks), None, "{}"

# Helper function to convert GeoJSON to DataFrame 
def convert_geojson_to_dataframe(geojson):
    # Check if the GeoJSON contains any features
    if 'features' not in geojson or len(geojson['features']) == 0:
        return pd.DataFrame()

    coords = []
    for feature in geojson['features']:
        geom = feature.get('geometry', None)
        # Check if geometry exists and is of type Point
        if geom and geom['type'] == 'Point':
            coords.append(geom['coordinates'])  # [lon, lat]
        else:
            print(f"Skipping non-Point or invalid feature: {geom}")

    if not coords:
        return pd.DataFrame()
    
    return pd.DataFrame(coords, columns=['xlong', 'ylat'])

# # Helper function to format predictions for display
# def format_predictions(predictions_df):
#     # Formatting the DataFrame as an HTML table or list
#     return html.Ul([html.Li(f"Longitude: {row['xlong']}, Latitude: {row['ylat']}, Prediction: {row['prediction']}")
#                     for index, row in predictions_df.iterrows()])

@app.callback(
    [
        Output("coords-display-container", "children", allow_duplicate=True),  # Update the display container
        Output("coords-json", "children", allow_duplicate=True),  # Update the JSON data
        Output('marker-layer', 'children', allow_duplicate=True),
        Output("loading-output", "children")
    ],
    [Input("edit_control", "geojson")],  # Listen for clicks on the submit button
    [State("coords-json", "children")],  # Maintain the state of coords-json
    prevent_initial_call=True   
)
def trigger_action_and_predict(geojson, json_coords):
    ctx = dash.callback_context

    if not ctx.triggered or "features" not in geojson:
        raise PreventUpdate

    randomForest = RandomForest()
    model = randomForest.load_model('random_forest_model.joblib')
    
    # Load existing coordinates from json_coords, if any
    #if json_coords and json_coords != "{}":
    #    df_coords = pd.read_json(json_coords, orient='split')
    #else:
    #    df_coords = pd.DataFrame(columns=["group", "#", "Type", "Latitude", "Longitude", "Prediction"])

    df_coords = pd.DataFrame(columns=["group", "#", "Type", "Latitude", "Longitude", "Prediction", "Wind Speed"])
    new_rows = []  # To hold new data from geojson

    current_group = 1  # Initialize group counter
    current_counter = 1  # Initialize counter for each group
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if geom_type == "Point" or geom_type == "Polygon":
            current_group, current_counter = process_geometry(geometry, new_rows,current_group, current_counter)

    # If there are new rows, predict and update df_coords
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        #Add an identifier
        new_df['temp_id'] = range(1, len(new_df) + 1)
        # Perform prediction for new rows
        predictions = model.predict_with_location(new_df[["group", "#", "Type", "Latitude", "Longitude", "Prediction", "temp_id"]])
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        predictions_df['temp_id'] = new_df['temp_id']
        
        new_df = pd.merge(new_df, predictions_df, on='temp_id', how='left')
        new_df.drop('temp_id', axis=1, inplace=True)  # Remove the temporary identifier
        
        #new_df['Prediction'] = predictions  # Update the DataFrame with new predictions
        df_coords = pd.concat([df_coords, new_df], ignore_index=True)
        
        df_coords['Prediction'] = df_coords['Prediction_y'].combine_first(df_coords['Prediction_x'])
        
        df_coords.drop(['Prediction_x', 'Prediction_y'], axis=1, inplace=True)

        #df_coords = pd.concat([df_coords, new_df], ignore_index=True)
        #df_coords = pd.merge(df_coords, new_df, 
        #                on=["group", "#", "Type", "Latitude", "Longitude"], 
        #                how='outer', 
        #                suffixes=('', '_new'))
        #print("after merge", df_coords)
        #df_coords['Prediction'] = df_coords['Prediction_new'].combine_first(df_coords['Prediction'])
        #df_coords.drop(['Prediction_new'], axis=1, inplace=True)
    
    table = setupDisplayTable(df_coords)

    # Convert updated df_coords to JSON for transmission
    updated_json_coords = df_coords.to_json(orient='split')     
    
    
    if not geojson or 'features' not in geojson:
        raise PreventUpdate

    # Initialize list to hold updated markers and polygons
    
    updated_features = []

    windSpeeds = []
    # Loop through each feature in the GeoJSON
    for feature in geojson['features']:
        geom_type = feature['geometry']['type']
        coords = feature['geometry']['coordinates']
        if geom_type == 'Point':
            # Handle Point geometry (Marker)
            lon, lat = coords
            matching_rows = df_coords[(df_coords['Latitude'] == lat) & (df_coords['Longitude'] == lon)]
            wind_speed = fetch_wind_speed(lat, lon)  # Assume this function fetches the wind speed as a string
            if not matching_rows.empty:
                # Assuming there's only one match, get the prediction value
                prediction = matching_rows.iloc[0]['Prediction']
                tooltip_text = f"Wind Speed: {wind_speed}, Collision Risk: {prediction:.2f}%"
            else:
                tooltip_text = f"Wind Speed: {wind_speed}"
                new_marker = dl.Marker(position=[lat, lon], children=[
                dl.Tooltip(f"Wind Speed: {wind_speed}")
            ])
            new_marker = dl.Marker(position=[lat, lon], children=[
                dl.Tooltip(tooltip_text)
            ])
            updated_features.append(new_marker)
        elif geom_type == 'Polygon':
            # Handle Polygon geometry
            # Convert GeoJSON coordinates to Leaflet polygon coordinates format
            polygon_coords = [[lat_lon[::-1] for lat_lon in coords[0]]]  # Assuming exterior ring only

            print("df_coords looks like this", df_coords)
            groupNumber = []
            # Iterate through each ring in the polygon
            for ring in polygon_coords:
                # Iterate through each coordinate pair in the current ring
                for lat, lon in ring:
                    matching_rows = df_coords[(df_coords['Latitude'] == lat) & (df_coords['Longitude'] == lon)]
                    if len(matching_rows) > 0:
                        groupNumber.append(matching_rows.iloc[0]['group'])
                    wind_speed = extract_average_wind_speed(fetch_wind_speed(lat, lon))  # Fetch wind speed for each coordinate pair
                    windSpeeds.append(wind_speed)
                    #avgWindSpeed = average_wind_speed(wind_speed)
                    #print("average wind speed", avgWindSpeed, lat, lon)
                    #windSpeeds.append(float(avgWindSpeed))  # Convert wind speed to float and add to the list

            prediction_value = None
            if(len(groupNumber) > 0):
                distinct_element = next(iter(set(groupNumber)))
                matching_rows = df_coords[df_coords['group'] == distinct_element]
                prediction_value = matching_rows['Prediction'].iloc[0]
                              
            if (len(windSpeeds) > 0):
                valid_speeds = [speed for speed in windSpeeds if speed is not None]
                average_wind_speed = sum(valid_speeds) / len(valid_speeds)   
                print("average wind speed is", average_wind_speed)       
            else:
                average_wind_speed = 0
            
            #for lon, lat in polygon_coords:
            #    wind_speed = fetch_wind_speed(lat, lon) 
            #    windSpeeds.append(wind_speed)

            polygon_label = f"Average Wind Speed: {round(average_wind_speed)} mph, Collision Risk: {prediction_value:.2f}%"  # Customize this label as needed
            new_polygon = dl.Polygon(
                positions=polygon_coords,
                children=[dl.Tooltip(polygon_label)],
                color="#007bff",
                fill=True,
                fillColor="#ADD8E6",
                fillOpacity=0.5,
            )
            updated_features.append(new_polygon)
    
    # markers = []
    # for index, row in updated_features.iterrows():
    #     if row['Type'] == 'Point':
    #         marker = dl.Marker(position=[row['Latitude'], row['Longitude']], children=[
    #             dl.Tooltip(f"Prediction: {row['Prediction']}%")
    #         ])
    #         markers.append(marker)
            
    

   
    
    return table, updated_json_coords, updated_features, None

def setupDisplayTable(df_coords):
    # Prepare the table for display
    if 'group' in df_coords.columns and '#' in df_coords.columns:
        df_display = df_coords.drop(columns=['group', '#'])
    else:
        df_display = df_coords
        
    # Check if 'Prediction' column exists before renaming
    if 'Prediction' in df_display.columns:
        df_display['Prediction'] = pd.to_numeric(df_display['Prediction'], errors='coerce')
        df_display = df_display.rename(columns={'Prediction': 'Collision Risk (%)'})
        
        # Format and display only the first row of each group with the prediction value, subsequent rows as blank
        first_rows = df_coords.groupby('group').head(1).index
        for index, row in df_display.iterrows():
            if index in first_rows:
                prediction_value = row['Collision Risk (%)']
                windSpeed = fetch_wind_speed(row['Latitude'], row['Longitude'])
                if pd.isna(prediction_value):
                    # If the prediction value is NaN, display 'Pending'
                    df_display.at[index, 'Collision Risk (%)'] = 'Pending'
                    df_display.at[index, 'Wind Speed'] = windSpeed
                else:
                    # Format the first row of each group with the percentage value
                    df_display.at[index, 'Collision Risk (%)'] = f"{prediction_value:.2f}%"
                    df_display.at[index, 'Wind Speed'] = windSpeed
            else:
                # Leave the collision risk percentage blank for non-first rows in each group
                df_display.at[index, 'Collision Risk (%)'] = ''
                df_display.at[index, 'Wind Speed'] = ''
        
    if 'Type' in df_display.columns:
        df_display = df_display.rename(columns={'Type': 'Marker Type'})
        
    if 'Latitude' in df_display.columns and 'Longitude' in df_display.columns:
        df_display[['Latitude', 'Longitude']] = df_display[['Latitude', 'Longitude']].round(6)
     
    # Create an empty list to hold the button groups
    button_groups = []

#unqiue rows 
    # Iterate through the DataFrame rows
    for idx, row in df_display.iterrows():
        # Create a div that contains both buttons for the current row using Font Awesome icons
        if not row['unique_group_id']:
            button_group = html.Div([
                html.Span(html.I(className="fa-solid fa-user"), id={'type': 'user', 'index': row['unique_group_id']}, n_clicks=0, style={'display': 'none'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px'})
        else:
            button_group = html.Div([
                html.Span(html.I(className="fa-solid fa-user"), id={'type': 'user', 'index': row['unique_group_id']}, n_clicks=0, style={'display': 'none'}),
                html.Span(html.I(className="bi bi-hand-thumbs-up"), id={'type': 'thumbs-up', 'index': row['unique_group_id']}, n_clicks=0),
                html.Span(html.I(className="bi bi-hand-thumbs-down"), id={'type': 'thumbs-down', 'index': row['unique_group_id']}, n_clicks=0)
            ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px'})
        
        # Add the button group to the list
        button_groups.append(button_group)

    # Add the list of button groups as a new column in the DataFrame
    df_display['Feedback'] = button_groups
    
    if 'unique_group_id' in df_display.columns:    
       df_display.drop('unique_group_id', axis=1, inplace=True)
    
    table = dbc.Table.from_dataframe(df_display, striped=True, bordered=True, hover=True)
   
    return table

def process_geometry(geometry, new_rows, current_group, current_counter):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Point":
        lat, lon = coords[1], coords[0]
        unique_group_id = f"{geom_type}_{current_group}_{current_counter}"
        windSpeed = fetch_wind_speed(lat, lon)
        row = create_row(current_group, current_counter, geom_type, lat, lon, unique_group_id, windSpeed)
        new_rows.append(row)
        current_group += 1  # Increment group for each new Point
    elif geom_type == "Polygon":
        # Process each vertex of the polygon (assuming the first ring for simplicity)
        for index, coord in enumerate(coords[0]):
            lat, lon = coord[1], coord[0]
            label = "Polygon" if index == 0 else ""
            windSpeed = fetch_wind_speed(lat, lon)
            unique_group_id = f"{label}_{current_group}_{current_counter}" if index == 0 else ""  # Construct unique_group_id for the first vertex or leave blank for others
            row = create_row(current_group, current_counter, label, lat, lon, unique_group_id, windSpeed)
            new_rows.append(row)
        current_group += 1
    return current_group, current_counter

def create_row(group, counter, label, lat, lon, unique_group_id, windSpeed):
    return {
        "group": group,
        "#": counter,
        "Type": label,
        "Latitude": lat,
        "Longitude": lon,
        "Prediction": "Pending",  # Placeholder for prediction,
        "Wind Speed": windSpeed,
        "unique_group_id": unique_group_id 
    }

@app.callback(
    Output('output-container-button', 'children'),
    Input('train-button-random-forest', 'n_clicks')
)
def btn_TrainModel(n_clicks):
    if n_clicks > 0:
        trainModel()
        return 'Model Trained Successfully'
    else:
        return 'Not Training'

def trainModel(): 
    df = pd.read_csv('datasets/dataset.csv')

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
    

# Helper function to convert GeoJSON to DataFrame
def convert_geojson_to_dataframe(geojson):
    # Implement conversion logic
    return pd.DataFrame()

def format_predictions(predictions):    
    # Assuming `predictions` is a DataFrame with the specified columns
    if predictions.empty:
        return "No predictions available"

    # The DataFrame is already in the desired format, so we directly convert it to a Dash Bootstrap table
    table = dbc.Table.from_dataframe(predictions, striped=True, bordered=True, hover=True, index=False)
    
    return table

# Callback to display coordinates directly from the edit_control's geojson data.
@app.callback(
    [
        Output("coords-display-container", "children"),
        Output("coords-json", "children")
    ],
    [
        Input("edit_control", "geojson")
    ]
)
def display_coords(geojson):
    if not geojson or "features" not in geojson:
        raise PreventUpdate
    rows = []  # List to hold row data
    point_counter, polygon_counter, pointGroup = 1, 1, 0  # Initialize counters
    
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")
        
        if geom_type == "Point":
            lat, lon = coords[1], coords[0]
            label = "Point"
            unique_group_id = f"{label}_{pointGroup}_{point_counter}"  # Construct unique_group_id
            rows.append({"group": pointGroup, "#": point_counter, "Type": label,"Latitude": lat, "Longitude": lon, "Prediction": "Pending", "unique_group_id": unique_group_id })
            point_counter += 1  # Increment point counter
            pointGroup += 1
            
        elif geom_type == "Polygon":
            # Only label the first vertex distinctly for each polygon
            for index, coord in enumerate(coords[0]):  # Assuming the first ring of coordinates for simplicity
                lat, lon = coord[1], coord[0]
                if index == 0:  # Label only the first vertex distinctly
                    label = "Polygon"
                else:
                    label = ""  # Subsequent vertices can have a generic label or be left blank
                unique_group_id = f"{label}_{pointGroup}_{polygon_counter}" if index == 0 else ""  # Construct unique_group_id for the first vertex or leave blank for others
                rows.append({"group": pointGroup, "#": polygon_counter, "Type": label, "Latitude": lat, "Longitude": lon, "Prediction": "Pending", "unique_group_id": unique_group_id })
            polygon_counter += 1  # Increment polygon counter after processing all vertices of a polygon
            pointGroup += 1
            
    #if not rows:
    #    return "No coordinates available"

    # Convert rows into a DataFrame and then into a dbc.Table
    df_coords = pd.DataFrame(rows)
    table = setupDisplayTable(df_coords)
    #table = dbc.Table.from_dataframe(df_coords, striped=True, bordered=True, hover=True)
     # Convert DataFrame to JSON for easy transmission
    json_coords = df_coords.to_json(orient='split')
    return table, json_coords

@app.callback(  
    Output('forecast-graph', 'figure', allow_duplicate=True),
    Output('model-status', 'children'),
    Input('model-status', 'children'),  # A dummy Input to trigger the page load callback
    prevent_initial_call=True
)
def update_graph_on_load(_):
    # Call the load_forecast function to get the figure
    figure, status_message = load_forecast(_)
    return figure, status_message

@app.callback(
    Output('train-status', 'children'),
    Input('train-button-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def train_arima_model(n_clicks):
    # Replace the following path with the path to your dataset
    data_path = 'datasets/turbines/detailed_wind_turbine_collisions_bk.csv'
    if n_clicks > 0:
        # Load your dataset
        data = pd.read_csv(data_path)
        
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        # Specify the column names (time column first)
        columns = ['Timestamp']
        
        # Initialize the forecasting object with your data and column names
        forecast_arima = CollisionForecastARIMA(data, columns)
        
        # Prepare the data
        forecast_arima.prepare_data()
        
        # Fit the ARIMA model (specify the order parameters according to your dataset)
        forecast_arima.fit_model(order=(1, 1, 1))
        
        # Save the model
        model_filename = 'arima_forecast_arima.joblib'
        forecast_arima.save_model(model_filename)
        print(f"Model saved to {model_filename}")
        
        return 'Model trained and saved successfully.'
    else:
        # Button has not been clicked, do not update anything
        raise PreventUpdate

    
# @app.callback(
#     [dash.dependencies.Output('forecast-graph', 'figure', allow_duplicate=True),
#      dash.dependencies.Output('model-status', 'children',  allow_duplicate=True)],
#     [dash.dependencies.Input('load-forecast-btn', 'n_clicks')],
#     prevent_initial_call=True
# )
# def update_forecast(n_clicks):
#     if n_clicks is None:
#         # Prevents the callback from firing on app load
#         return go.Figure(), ''
#     else:
#         return load_forecast(n_clicks)

# @app.callback(
#     Output('forecast-graph', 'figure'),  # ID of the graph to update
#     [Input('forecast-period-slider', 'value')]  # ID of the slider component and property to listen to
# )
# def update_forecast_plot(forecast_period_slider_value):
#     try:
#         # Load the detailed collision data
#         detailed_collisions_path = 'exported_data.csv'
#         detailed_collisions_df = pd.read_csv(detailed_collisions_path)

#         # Load the wind turbine location data
#         wind_turbines_path = 'datasets/turbines/wind_turbines_with_collisions.csv'
#         wind_turbines_df = pd.read_csv(wind_turbines_path)

#         # Merge the datasets on longitude and latitude
#         merged_df = pd.merge(detailed_collisions_df, wind_turbines_df, 
#                              left_on=['Turbine_Longitude', 'Turbine_Latitude'], 
#                              right_on=['xlong', 'ylat'], 
#                              how='inner')

#         # Convert 'Timestamp' to datetime
#         merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], errors='coerce')

#         # Aggregate collision counts by date for the merged dataset
#         aggregated_data = merged_df.resample('D', on='Timestamp').agg({'collision': 'sum'}).reset_index()
#         aggregated_data.rename(columns={'Timestamp': 'ds', 'collision': 'y'}, inplace=True)

#         # Initialize and fit the Prophet model
#         model = Prophet()
#         model.fit(aggregated_data)

#         # Adjust the forecast period based on the slider's value
#         future_dates = model.make_future_dataframe(periods=forecast_period_slider_value)

#         # Predict the values for future dates
#         forecast = model.predict(future_dates)

#         # Create a figure to plot the forecast
#         fig = go.Figure()

#         # Plot the historical aggregated collision counts
#         fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Aggregated Collisions'))

#         # Add the forecasted data
#         fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))

#         # Update plot layout
#         fig.update_layout(
#             title='Aggregated Collision Counts and Forecast',
#             xaxis_title='Date',
#             yaxis_title='Number of Collisions',
#             xaxis_rangeslider_visible=True,
#             showlegend=True,
#             template="none"
#         )

#         return fig
#     except FileNotFoundError as e:
#         print(e)
#         return go.Figure(), 'Required file not found. Please check the file paths.'
from dash.dependencies import Input, Output

@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-period-slider', 'value'),
     Input('start-year-slider', 'value'),
     Input('init-trigger', 'n_intervals'),
     Input('coords-json', 'children')]
)
def update_forecast_plot(forecast_period_slider_value, start_year_slider_value, n, coords_json):
    # Check if coords_json is not provided and skip update if so
    if not coords_json or coords_json == "null":
        print("No new coordinates provided. Using only historical data.")
        raise PreventUpdate

    # Print contents of coords_json to console for debugging
    
    try:
        # Load the detailed collision data
        #detailed_collisions_path = 'exported_data.csv'
        #detailed_collisions_df = pd.read_csv(detailed_collisions_path)
        
        # Convert 'Timestamp' to datetime and filter data based on selected start year
        detailed_collisions_df['Timestamp'] = pd.to_datetime(detailed_collisions_df['Timestamp'], errors='coerce')
        start_date = pd.to_datetime(f"{start_year_slider_value}-01-01")
        filtered_df = detailed_collisions_df[detailed_collisions_df['Timestamp'] >= start_date]
        
        # Assuming each row represents a collision, count collisions by year
        filtered_df['Year'] = filtered_df['Timestamp'].dt.year
        aggregated_data = filtered_df.groupby('Year').size().reset_index(name='y')
        aggregated_data.rename(columns={'Year': 'ds'}, inplace=True)
        
        # Convert 'ds' from year to datetime format for Prophet
        aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'].astype(str) + '-01-01')

        print("before", aggregated_data)

        coords_data = json.loads(coords_json)

        # If coords_json is provided, adjust aggregated_data with additional_collisions
        if 'data' in coords_data and coords_data['data']:
            new_points_df = pd.read_json(coords_json, orient='split')
            new_points_df['Prediction'].replace('Pending', np.nan, inplace=True)
            today_date = datetime.today().strftime('%Y-%m-%d')
            #new_points_df['Prediction'] = pd.to_numeric(new_points_df['Prediction'], errors='coerce')
            additional_collisions = new_points_df['Prediction'].sum()
            
            # Convert 'ds' column to datetime for comparison
            aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'])
            #print("after", aggregated_data)
            # Check if today's date exists in 'ds' column and update or append accordingly
            if pd.to_datetime(today_date) in aggregated_data['ds'].values:
                print("added new collision value")
                aggregated_data.loc[aggregated_data['ds'] == pd.to_datetime(today_date), 'y'] += additional_collisions
                
            else:
                print("added new collision value")
                new_row = {'ds': pd.to_datetime(today_date), 'y': additional_collisions}
                new_row_df = pd.DataFrame([new_row])
                aggregated_data = pd.concat([aggregated_data, new_row_df], ignore_index=True)

        # Prophet forecasting
        model = Prophet()
        model.fit(aggregated_data)
        future_dates = model.make_future_dataframe(periods=forecast_period_slider_value)
        forecast = model.predict(future_dates)

        newest_date = aggregated_data['ds'].max()

        # Create and update the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Aggregated Collisions'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))
        fig.update_layout(
            title='Yearly Collision Counts and Forecast',
            xaxis_title='Year',
            yaxis_title='Number of Collisions',
            xaxis_rangeslider_visible=True,
            showlegend=True,
            template="none"
           # xaxis=dict(
           #     range=[min_year, max_year],  # replace 'start_year' with the actual start year
           # ),
        )


        return fig
    except FileNotFoundError as e:
        print(e)
        return go.Figure(), 'Required file not found. Please check the file paths.'

# Define callback to handle the thumbs button clicks
@app.callback(
    Output('model-feedback-container', 'children'),
    [Input({'type': 'user', 'index': ALL}, 'n_clicks'),
     Input({'type': 'thumbs-up', 'index': ALL}, 'n_clicks'),
     Input({'type': 'thumbs-down', 'index': ALL}, 'n_clicks')],
    [State('coords-json', 'children')],
    prevent_initial_call=True
)
def handle_thumbs_clicks(*args):
    
    ctx = callback_context
    state = ctx.states
    coords_json = state['coords-json.children']
    
    if not coords_json or not ctx.triggered:
         return ""

    button_info = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    idx, type = button_info['index'], button_info['type']
    button_type = button_info['type']
    
    cords = []    
    if 'Polygon' in idx:
        parts = idx.split('_')
        coords_json = json.loads(coords_json)

        # Convert the 'data' list into a list of dictionaries for easier searching
        columns = coords_json['columns']
        data_rows = coords_json['data']
        data_dicts = [dict(zip(columns, row)) for row in data_rows]
        
        # Search for the matching unique_group_id
        for row in data_dicts:
            if row.get('group') == int(parts[1]) and row.get('#') == int(parts[2]):
                latitude = row.get('Latitude')
                longitude = row.get('Longitude')
                prediction = row.get('Prediction')
                cords.append((latitude, longitude, prediction))  # Append as a tuple
    else:
        latitude, longitude, prediction = find_coords_by_unique_group_id(coords_json, idx)
        cords.append((latitude, longitude, prediction))  # Append as a tuple
  
    response = ""
    if type == 'user':
        #response = f"User interaction recorded for user"
        response = None
    elif type == 'thumbs-up':
        for latitude, longitude, prediction in cords:
            if (float(prediction) > 0.0):
                value = 1
            else:
                value = 0
            update_collision_in_csv(latitude, longitude, value)
            response = f"Collision value updated for Latitude: {latitude}, Longitude: {longitude}"
            #response = None
    elif type == 'thumbs-down':
        value = -1
        for latitude, longitude, prediction in cords:
            update_collision_in_csv(latitude, longitude, value)
            response = f"Collision value updated for Latitude: {latitude}, Longitude: {longitude}"
            #response = None
    else:
        raise ValueError("Invalid interaction type.")  # Handle unexpected interaction types
    
    return response

def find_coords_by_unique_group_id(coords_json_str, unique_group_id):
    # Load the JSON string into a Python dictionary
    coords_json = json.loads(coords_json_str)
    
    # Convert the 'data' list into a list of dictionaries for easier searching
    columns = coords_json['columns']
    data_rows = coords_json['data']
    data_dicts = [dict(zip(columns, row)) for row in data_rows]
    
    # Search for the matching unique_group_id
    for row in data_dicts:
        if row.get('unique_group_id') == unique_group_id:
            latitude = row.get('Latitude')
            longitude = row.get('Longitude')
            prediction = row.get('Prediction')
            return latitude, longitude, prediction
            
    # Return None if no match found
    return None, None, None

def update_collision_in_csv(latitude, longitude, adjustment_value):
    # Load the CSV file
    csv_file_path = 'datasets/turbines/wind_turbines_with_collisions.csv'
    df = pd.read_csv(csv_file_path)
    
    # Find the row with the matching latitude and longitude
    match = df[(df['xlong'] == longitude) & (df['ylat'] == latitude)]

    if not match.empty:
        # If there's a match, update the collision value
        for index, row in match.iterrows():
            new_collision_value = max(0, row['collision'] + adjustment_value)  # Ensure collision doesn't go below 0
            df.at[index, 'collision'] = new_collision_value
    else:
        # If no match, create a new record with the collision value adjusted from 0
        new_collision_value = max(0, 0 + adjustment_value)  # Starting from 0, adjust by the adjustment_value but ensure it doesn't go below 0
        new_record = pd.DataFrame({'xlong': [longitude], 'ylat': [latitude], 'collision': [new_collision_value]})
        df = pd.concat([df, new_record], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_file_path, index=False)
    return True

@app.callback(
    [Output({'type': 'thumbs-down', 'index': MATCH}, 'id'),
     Output({'type': 'thumbs-down', 'index': MATCH}, 'children'),
     Output({'type': 'thumbs-down', 'index': MATCH}, 'className')],
    [Input({'type': 'thumbs-down', 'index': ALL}, 'n_clicks')],
    [State({'type': 'thumbs-up', 'index': MATCH}, 'className')],
    prevent_initial_call=True
)
def update_icon(n_clicks_list, thumbs_down_class):
    ctx = dash.callback_context

    if not ctx.triggered:
        return [dash.no_update, dash.no_update, dash.no_update]  # Prevent update if no buttons were clicked

    # Determine which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_info = json.loads(button_id)
    idx = button_info['index']   
 
    total_clicks = sum(n_clicks_list)
    if total_clicks % 2 == 1:
        new_id = {'type': 'thumbs-down', 'index': idx}
        new_content = html.I(className="bi bi-hand-thumbs-down-fill")  # Example new content
        new_class = ''  # Resetting the class
    else:
        new_id = {'type': 'thumbs-down', 'index': idx}
        new_content = html.I(className="bi bi-hand-thumbs-down")  # Example new content
        new_class = ''  # Resetting the class

    return [new_id, new_content, new_class]


@app.callback(
    [Output({'type': 'thumbs-up', 'index': MATCH}, 'id'),
     Output({'type': 'thumbs-up', 'index': MATCH}, 'children'),
     Output({'type': 'thumbs-up', 'index': MATCH}, 'className')],
    [Input({'type': 'thumbs-up', 'index': ALL}, 'n_clicks')],
    [State({'type': 'thumbs-down', 'index': MATCH}, 'className')],
    prevent_initial_call=True
)
def update_icon(n_clicks_list, thumbs_down_class):
    ctx = dash.callback_context

    print("thumbs down class is", thumbs_down_class)
    if not ctx.triggered:
        return [dash.no_update, dash.no_update, dash.no_update]  # Prevent update if no buttons were clicked

    # Determine which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_info = json.loads(button_id)
    idx = button_info['index']   
 
    total_clicks = sum(n_clicks_list)
    if total_clicks % 2 == 1:
        new_id = {'type': 'thumbs-up', 'index': idx}
        new_content = html.I(className="bi bi-hand-thumbs-up-fill")  # Example new content
        new_class = ''  # Resetting the class
    else:
        new_id = {'type': 'thumbs-up', 'index': idx}
        new_content = html.I(className="bi bi-hand-thumbs-up")  # Example new content
        new_class = ''  # Resetting the class

    return [new_id, new_content, new_class]

# Function to fetch wind speed from the National Weather Service API
def fetch_wind_speed(lat, lon):
    url = f"https://api.weather.gov/points/{lat},{lon}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        forecast_url = response.json()['properties']['forecast']
        forecast_response = requests.get(forecast_url, headers={'User-Agent': 'Mozilla/5.0'})
        current_forecast = forecast_response.json()['properties']['periods'][0]
        wind_speed = current_forecast['windSpeed']
        return wind_speed
    except Exception as e:
        print(f"Error fetching wind speed: {e}")
        return "Error"

# @app.callback(
#     Output('marker-layer', 'children'),
#     [Input("edit_control", "geojson"),
#      Input("marker-layer", "geojson")
#      ],
    
#     #[State('marker-layer', 'children')]
# )
# def update_markers(geojson_data, existing_markers):
#     if not geojson_data or 'features' not in geojson_data:
#         raise PreventUpdate

#     # Initialize list to hold updated markers and polygons
#     updated_features = existing_markers if existing_markers else []

#     windSpeeds = []
#     # Loop through each feature in the GeoJSON
#     for feature in geojson_data['features']:
#         geom_type = feature['geometry']['type']
#         coords = feature['geometry']['coordinates']

#         if geom_type == 'Point':
#             # Handle Point geometry (Marker)
#             lon, lat = coords
#             wind_speed = fetch_wind_speed(lat, lon)  # Assume this function fetches the wind speed as a string
#             #new_marker = dl.Marker(position=[lat, lon], children=[
#             #    dl.Popup(content=f"This is <b>Wind Speed: {wind_speed}</b>!")
#             #])
#             #icon = {'className': 'custom-icon', 'html': f'<b>{wind_speed} mph</b>'}  # Example icon definition
#             #new_marker = dl.DivMarker(position=[lat, lon], iconOptions=icon)
#             new_marker = dl.Marker(position=[lat, lon], children=[
#                 dl.Popup(content=f"This is <b>Wind Speed: {wind_speed}</b>!")
#             ])
#             updated_features.append(new_marker)
#         elif geom_type == 'Polygon':
#             # Handle Polygon geometry
#             # Convert GeoJSON coordinates to Leaflet polygon coordinates format
#             polygon_coords = [[lat_lon[::-1] for lat_lon in coords[0]]]  # Assuming exterior ring only
           

#             # Iterate through each ring in the polygon
#             for ring in polygon_coords:
#                 # Iterate through each coordinate pair in the current ring
#                 for lat, lon in ring:
#                     wind_speed = extract_average_wind_speed(fetch_wind_speed(lat, lon))  # Fetch wind speed for each coordinate pair
#                     windSpeeds.append(wind_speed)
#                     #avgWindSpeed = average_wind_speed(wind_speed)
#                     #print("average wind speed", avgWindSpeed, lat, lon)
#                     #windSpeeds.append(float(avgWindSpeed))  # Convert wind speed to float and add to the list



#             if (len(windSpeeds) > 0):
#                 valid_speeds = [speed for speed in windSpeeds if speed is not None]
#                 average_wind_speed = sum(valid_speeds) / len(valid_speeds)   
#                 print("average wind speed is", average_wind_speed)       
#             else:
#                 average_wind_speed = 0
            
#             #for lon, lat in polygon_coords:
#             #    wind_speed = fetch_wind_speed(lat, lon) 
#             #    windSpeeds.append(wind_speed)

#             polygon_label = f"Average Wind Speed: {round(average_wind_speed)} mph"  # Customize this label as needed
#             new_polygon = dl.Polygon(
#                 positions=polygon_coords,
#                 children=[dl.Popup(content=f"This is <b>{polygon_label}</b>!")],
#                 color="#007bff",
#                 fill=True,
#                 fillColor="#ADD8E6",
#                 fillOpacity=0.5,
#             )
#             updated_features.append(new_polygon)

#     return updated_features

def average_wind_speed(wind_speeds):
    """
    Calculate the average wind speed from a list of wind speeds.

    Parameters:
    wind_speeds (list of float): List of wind speeds in mph.

    Returns:
    float: The average wind speed.
    """
    if not wind_speeds:  # Check if the list is empty
        return 0.0  # Return 0 or any default value you deem appropriate
    return sum(wind_speeds) / len(wind_speeds)

def extract_average_wind_speed(wind_speed_str):
    # Find all numbers in the string
    numbers = re.findall(r'\d+', wind_speed_str)
    # Convert found strings to floats
    numbers = [float(num) for num in numbers]
    # Calculate the average if there are any numbers
    if numbers:
        return sum(numbers) / len(numbers)
    else:
        return None  # Or some default value



# Trigger mode (edit) + action (remove all)
@app.callback(Output("edit_control", "editToolbar"), Input("btn", "n_clicks"))
def trigger_action(n_clicks):   
    return dict(mode="remove", action="clear all", n_clicks=n_clicks)  # include n_click to ensure prop changes




# @app.callback(
#     Output("marker-layer", "children"),
#     [Input("map", "geojson")],
# )
# def update_markers(geojson):
#     print("in update markers")
#     # Parse the GeoJSON to update markers
#     if not geojson or not geojson['features']:
#         return []  # Return an empty list if there are no features

#     markers = []
#     for feature in geojson['features']:
#         if feature['geometry']['type'] == 'Point':
#             coords = feature['geometry']['coordinates']
#             marker = dl.Marker(position=[coords[1], coords[0]])
#             markers.append(marker)
    
#     return []


if __name__ == "__main__":
    app.run_server(debug=True, port=8060)
    print(os.getcwd())
    print("Current Working Directory:", os.getcwd())

#if __name__=="__main__":
 # app.run(host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']))
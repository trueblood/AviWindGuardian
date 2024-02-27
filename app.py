# -*- coding: utf-8 -*-
import json
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
from scripts.randomforest import RandomForest
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from scripts.collisionforecastarima import CollisionForecastARIMA
import warnings
from datetime import datetime, timedelta
from prophet import Prophet

warnings.filterwarnings("ignore")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME],
)

"""
==========================================================================
Markdown Text
"""
learn_text = dcc.Markdown(
    """
    Past performance certainly does not determine future results, but you can still
    learn a lot by reviewing how various asset classes have performed over time.

    Use the sliders to change the asset allocation (how much you invest in cash vs
    bonds vs stock) and see how this affects your returns.

    Note that the results shown in "My Portfolio" assumes rebalancing was done at
    the beginning of every year.  Also, this information is based on the S&P 500 index
    as a proxy for "stocks", the 10 year US Treasury Bond for "bonds" and the 3 month
    US Treasury Bill for "cash."  Your results of course,  would be different based
    on your actual holdings.

    This is intended to help you determine your investment philosophy and understand
    what sort of risks and returns you might see for each asset category.

    The  data is from [Aswath Damodaran](http://people.stern.nyu.edu/adamodar/New_Home_Page/home.htm)
    who teaches  corporate finance and valuation at the Stern School of Business
    at New York University.

    Check out his excellent on-line course in
    [Investment Philosophies.](http://people.stern.nyu.edu/adamodar/New_Home_Page/webcastinvphil.htm)
    """
)

footer = html.Div(
    dcc.Markdown(
        """
         This information is intended solely as general information for educational
        and entertainment purposes only and is not a substitute for professional advice and
        services from qualified financial services providers familiar with your financial
        situation.    
        """
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
        dl.FeatureGroup([
            dl.EditControl(
                id="edit_control", 
                position="topright",
                draw={
                    "polyline": False,  # Disable line drawing
                    "polygon": True,    # Keep polygon drawing
                    "circle": False,    # Disable circle drawing
                    "rectangle": False, # Disable rectangle drawing
                    "marker": True,     # Enable marker drawing
                    "circlemarker": False # Disable circlemarker drawing
                }
            )
        ])
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
def load_forecast(_):
    #print("in load forecast")
    try:
        # Load the detailed collision data
        detailed_collisions_path = 'exported_data.csv'
        detailed_collisions_df = pd.read_csv(detailed_collisions_path)
        #print(detailed_collisions_df.head())
        # Load the wind turbine location data
        wind_turbines_path = 'datasets/turbines/wind_turbines_with_collisions.csv'
        wind_turbines_df = pd.read_csv(wind_turbines_path)
        #print(wind_turbines_df.head())
        # Merge the datasets on longitude and latitude
        merged_df = pd.merge(detailed_collisions_df, wind_turbines_df, 
                     left_on=['Turbine_Longitude', 'Turbine_Latitude'], 
                     right_on=['xlong', 'ylat'], 
                     how='inner')
        
        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], errors='coerce')
        # Assuming 'Timestamp' column exists and represents when collisions occurred
        #print(merged_df.head()) 
        
        # Aggregate collision counts by date for the merged dataset
        aggregated_data = merged_df.resample('D', on='Timestamp').agg({'collision': 'sum'}).reset_index()
        aggregated_data.rename(columns={'Timestamp': 'ds', 'collision': 'y'}, inplace=True)
        
        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(aggregated_data)
        #print("after prophet fit")
        
        # Create future dataframe for forecasting (e.g., next 365 days)
        future_dates = model.make_future_dataframe(periods=365)
        
        # Predict the values for future dates
        forecast = model.predict(future_dates)
        
        # Create a figure to plot the forecast
        fig = go.Figure()
        
        # Plot the historical aggregated collision counts
        fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Aggregated Collisions'))
        
        # Add the forecasted data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))
        
        # Update plot layout
        fig.update_layout(
            title='Aggregated Collision Counts and Forecast',
            xaxis_title='Date',
            yaxis_title='Number of Collisions',
            xaxis_rangeslider_visible=True,
            showlegend=True,
            template="none"
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
            html.Div(id="coords-display-container", style={'overflowY': 'scroll', 'height': '50vh'}),
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
                        value=90,  # Default to 90 days
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
                        marks={i: str(i) for i in range(2011, 2023)},  # Example range
                        min=2011,
                        max=2022,
                        step=1,
                        value=2011,  # Default to 2021
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
        html.Button('Train Random Forest Model', id='train-button-random-forest', n_clicks=0),
        html.Button('Train Forecasting Model', id='train-button-forecast', n_clicks=0),
        html.Div(id='output-container-button'),
        html.Div(id='train-status'),
        html.Div(id='model-status')
    ],
    className="mt-4",
)

# ========= Learn Tab  Components
learn_card = dbc.Card(
    [
        dbc.CardHeader("An Introduction to Asset Allocation"),
        dbc.CardBody(learn_text),
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
            label="Play",
            className="pb-4",
        ),
        dbc.Tab([model_training_card], tab_id="tab-3", label="Training")
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
                        dcc.Graph(id='forecast-graph', className="pb-4", figure=load_forecast(1)[0]),
                        html.Div(id="prediction-output"),
                        html.Div(id='coords-json', style={'display': 'none'}),
                        dcc.Interval(
                            id='init-trigger',
                            interval=1,  # in milliseconds
                            n_intervals=0,
                            max_intervals=1  # Stop after the first call
                        )
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
        print("No features in GeoJSON")
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
        print("No valid Point coordinates found in GeoJSON features")
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
        Output("coords-json", "children", allow_duplicate=True)  # Update the JSON data
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

    df_coords = pd.DataFrame(columns=["group", "#", "Type", "Latitude", "Longitude", "Prediction"])
    new_rows = []  # To hold new data from geojson

    current_group = 1  # Initialize group counter
    current_counter = 1  # Initialize counter for each group
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if geom_type == "Point" or geom_type == "Polygon":
            print("in point or polygon process geometry")
            current_group, current_counter = process_geometry(geometry, new_rows,current_group, current_counter)

    # If there are new rows, predict and update df_coords
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        print("New rows from GeoJSON:")
        #Add an identifier
        new_df['temp_id'] = range(1, len(new_df) + 1)
        print("New rows from GeoJSON:", new_df.head())
        # Perform prediction for new rows
        predictions = model.predict_with_location(new_df[["group", "#", "Type", "Latitude", "Longitude", "Prediction", "temp_id"]])
        print("after predictions", predictions)
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        predictions_df['temp_id'] = new_df['temp_id']
        
        new_df = pd.merge(new_df, predictions_df, on='temp_id', how='left')
        new_df.drop('temp_id', axis=1, inplace=True)  # Remove the temporary identifier
        print("After attaching predictions to new_df", new_df)
        
        #new_df['Prediction'] = predictions  # Update the DataFrame with new predictions
        print("after new_df", new_df)
        print("before merge", new_df.head())
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
        print("Updated df_coords with new predictions", df_coords)
        
    print("df_coords", df_coords)
    

    table = setupDisplayTable(df_coords)
    print("after table create")

    # Convert updated df_coords to JSON for transmission
    updated_json_coords = df_coords.to_json(orient='split')
    print("Here are updated cords", updated_json_coords)
    return table, updated_json_coords

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
                if pd.isna(prediction_value):
                    # If the prediction value is NaN, display 'Pending'
                    df_display.at[index, 'Collision Risk (%)'] = 'Pending'
                else:
                    # Format the first row of each group with the percentage value
                    df_display.at[index, 'Collision Risk (%)'] = f"{prediction_value:.2f}%"
            else:
                # Leave the collision risk percentage blank for non-first rows in each group
                df_display.at[index, 'Collision Risk (%)'] = ''
        
    if 'Type' in df_display.columns:
        df_display = df_display.rename(columns={'Type': 'Marker Type'})
        
    if 'Latitude' in df_display.columns and 'Longitude' in df_display.columns:
        df_display[['Latitude', 'Longitude']] = df_display[['Latitude', 'Longitude']].round(6)

    # Create an empty list to hold the button groups
    button_groups = []

    # Iterate through the DataFrame rows
    for idx, row in df_display.iterrows():
        # Create a div that contains both buttons for the current row using Font Awesome icons
        button_group = html.Div([
            html.Button(html.I(className="fas fa-thumbs-up"), id={'type': 'thumbs-up', 'index': idx}, n_clicks=0),
            html.Button(html.I(className="fas fa-thumbs-down"), id={'type': 'thumbs-down', 'index': idx}, n_clicks=0)
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px'})
        
        # Add the button group to the list
        button_groups.append(button_group)

    # Add the list of button groups as a new column in the DataFrame
    df_display['Buttons'] = button_groups
    
    table = dbc.Table.from_dataframe(df_display, striped=True, bordered=True, hover=True)
   
    return table

def process_geometry(geometry, new_rows, current_group, current_counter):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Point":
        lat, lon = coords[1], coords[0]
        row = create_row(current_group, current_counter, geom_type, lat, lon)
        new_rows.append(row)
        current_group += 1  # Increment group for each new Point
    elif geom_type == "Polygon":
        # Process each vertex of the polygon (assuming the first ring for simplicity)
        for index, coord in enumerate(coords[0]):
            lat, lon = coord[1], coord[0]
            label = "Polygon" if index == 0 else ""
            row = create_row(current_group, current_counter, label, lat, lon)
            new_rows.append(row)
        current_group += 1
    return current_group, current_counter

def create_row(group, counter, label, lat, lon):
    return {
        "group": group,
        "#": counter,
        "Type": label,
        "Latitude": lat,
        "Longitude": lon,
        "Prediction": "Pending"  # Placeholder for prediction
    }

@app.callback(
    Output('output-container-button', 'children'),
    Input('train-button-random-forest', 'n_clicks')
)
def btn_TrainModel(n_clicks):
    if n_clicks > 0:
        trainModel()
        return 'Model trained.'
    else:
        return 'Button not clicked yet.'

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
    randomForest = RandomForest()

    mean_accuracy, std_dev = randomForest.evaluate_model_with_kfold(X, Y, n_splits=5)
    print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_dev}")
    

# Helper function to convert GeoJSON to DataFrame
def convert_geojson_to_dataframe(geojson):
    # Implement conversion logic
    return pd.DataFrame()

def format_predictions(predictions):
    
    print("In format predictions")
    print(predictions)
    
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
            rows.append({"group": pointGroup, "#": point_counter, "Type": label,"Latitude": lat, "Longitude": lon, "Prediction": "Pending"})
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
                rows.append({"group": pointGroup, "#": polygon_counter, "Type": label, "Latitude": lat, "Longitude": lon, "Prediction": "Pending"})
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
    print("in update_graph_on_load")
    # Call the load_forecast function to get the figure
    figure, status_message = load_forecast(_)
    return figure, status_message

@app.callback(
    Output('train-status', 'children'),
    Input('train-button-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def train_arima_model(n_clicks):
    print("in train arima model")
    # Replace the following path with the path to your dataset
    data_path = 'datasets/turbines/detailed_wind_turbine_collisions_bk.csv'
    if n_clicks > 0:
        # Load your dataset
        data = pd.read_csv(data_path)
        
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        print(data)
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
    print("in update_forecast_plot from page laod")
    # Check if coords_json is not provided and skip update if so
    if not coords_json or coords_json == "null":
        print("No new coordinates provided. Using only historical data.")
        raise PreventUpdate

    # Print contents of coords_json to console for debugging
    print(f"Received coords_json contents: {coords_json}")
    
    try:
        # Load and prepare your datasets
        detailed_collisions_path = 'exported_data.csv'
        wind_turbines_path = 'datasets/turbines/wind_turbines_with_collisions.csv'
        detailed_collisions_df = pd.read_csv(detailed_collisions_path)
        wind_turbines_df = pd.read_csv(wind_turbines_path)
        
        # Merge datasets
        merged_df = pd.merge(detailed_collisions_df, wind_turbines_df, 
                             left_on=['Turbine_Longitude', 'Turbine_Latitude'], 
                             right_on=['xlong', 'ylat'], 
                             how='inner')
        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], errors='coerce')

        # Filter data based on selected start year for demonstration purposes
        start_date = pd.to_datetime(f"{start_year_slider_value}-01-01")
        filtered_df = merged_df[merged_df['Timestamp'] >= start_date]

        # Aggregate collision counts by date
        aggregated_data = filtered_df.resample('D', on='Timestamp').agg({'collision': 'sum'}).reset_index()
        aggregated_data.rename(columns={'Timestamp': 'ds', 'collision': 'y'}, inplace=True)

        # Parse coords_json to check if it's effectively empty
        coords_data = json.loads(coords_json)
        #if 'data' in coords_data and not coords_data['data']:
        #    print("Received empty dataset in coords_json. Skipping processing for new coordinates.")
        #    raise PreventUpdate

        # If coords_json is provided, adjust aggregated_data with additional_collisions
        if 'data' in coords_data and coords_data['data']:
            print("in coords_json and coords_json")
            new_points_df = pd.read_json(coords_json, orient='split')
            new_points_df['Prediction'].replace('Pending', np.nan, inplace=True)
            print("new points", new_points_df)
            today_date = datetime.today().strftime('%Y-%m-%d')
            #new_points_df['Prediction'] = pd.to_numeric(new_points_df['Prediction'], errors='coerce')
            additional_collisions = new_points_df['Prediction'].sum()
            
            # Convert 'ds' column to datetime for comparison
            aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'])
            #print("after", aggregated_data)
            print("before if")
            # Check if today's date exists in 'ds' column and update or append accordingly
            if pd.to_datetime(today_date) in aggregated_data['ds'].values:
                print("in if")
                aggregated_data.loc[aggregated_data['ds'] == pd.to_datetime(today_date), 'y'] += additional_collisions
                print("added new collision points to existing date")
                
            else:
                print("in else")
                new_row = {'ds': pd.to_datetime(today_date), 'y': additional_collisions}
                print("new row", new_row)
                new_row_df = pd.DataFrame([new_row])
                aggregated_data = pd.concat([aggregated_data, new_row_df], ignore_index=True)
                print("added new collision points to existing date")

        print("before getting model")
        # Prophet forecasting
        model = Prophet()
        model.fit(aggregated_data)
        print("after fitting model")
        future_dates = model.make_future_dataframe(periods=forecast_period_slider_value)
        forecast = model.predict(future_dates)

        newest_date = aggregated_data['ds'].max()
        print(newest_date)

        # Create and update the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aggregated_data['ds'], y=aggregated_data['y'], mode='lines', name='Historical Aggregated Collisions'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecasted Collisions', line=dict(dash='dot')))
        fig.update_layout(title='Aggregated Collision Counts and Forecast',
                          xaxis_title='Date', yaxis_title='Number of Collisions',
                          xaxis_rangeslider_visible=True, showlegend=True, template="none")
        print("figure updated")

        return fig
    except FileNotFoundError as e:
        print(e)
        return go.Figure(), 'Required file not found. Please check the file paths.'


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    print(os.getcwd())
    print("Current Working Directory:", os.getcwd())

#if __name__=="__main__":
 # app.run(host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']))
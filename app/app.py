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
from src.randomforest import RandomForest
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import warnings
from datetime import datetime
from prophet import Prophet
from dash.dependencies import Input, Output, State, MATCH, ALL
import requests
import re
from dash.dependencies import Input, Output

warnings.filterwarnings("ignore")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP],
)

# Load the detailed collision data
detailed_collisions_path = os.path.join(os.getcwd(), 'src/datasets/turbines/detailed_wind_turbine_collisions.csv') 
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

def load_forecast():
    try:
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
        )

        return fig, 'Model loaded and forecast generated successfully.'
    except FileNotFoundError as e:
        print(e)
        return go.Figure(), 'Required file not found. Please check the file paths.'

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

slider_card_forecast = dbc.Card(
    [
        html.H4("Adjust Forecast Parameters:", className="card-title"),
        html.Div([
            dbc.Card(
                [
                    html.H5("Forecast Period (Days):", className="card-title"),
                    dcc.Slider(
                        id="forecast-period-slider",
                        marks={i: f"{i} days" for i in range(0, 366, 60)},  # Adjust based on your needs
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
            html.Br(),
            html.Div(id='output-container-button'),
            # Training Status Table
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Machine Learning Models"), html.Th("Status")]))] +
                [html.Tbody([html.Tr([html.Td(dbc.Button("Train Random Forest Model", id='train-button-random-forest', n_clicks=0, color="primary", className="me-1, btn-table-width")), html.Td(html.Div(id='train-status', children="Not Training"))]),
                             html.Tr([html.Td(dbc.Button("Train Forecasting Model", id='train-button-forecast', n_clicks=0, color="primary", className="me-1, btn-table-width")), html.Td(html.Div(id='model-status', children="Not Training"))])])],
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
                        dcc.Graph(id='forecast-graph', className="pb-4", figure=load_forecast()[0],
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
        
        df_coords = pd.concat([df_coords, new_df], ignore_index=True)
        df_coords['Prediction'] = df_coords['Prediction_y'].combine_first(df_coords['Prediction_x'])
        df_coords.drop(['Prediction_x', 'Prediction_y'], axis=1, inplace=True)
        
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
            
    # Convert rows into a DataFrame and then into a dbc.Table
    df_coords = pd.DataFrame(rows)
    table = setupDisplayTable(df_coords)
    json_coords = df_coords.to_json(orient='split')
    return table, json_coords

@app.callback(  
    Output('forecast-graph', 'figure', allow_duplicate=True),
    Output('model-status', 'children'),
    Input('train-button-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def update_graph_on_load(n_clicks):
    if (n_clicks > 0):
        figure, status_message = load_forecast()
        return figure, status_message

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
    try:
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
            additional_collisions = new_points_df['Prediction'].sum()
            
            # Convert 'ds' column to datetime for comparison
            aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'])
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
    csv_file_path = os.path.join(os.getcwd(), 'src/datasets/turbines/wind_turbines_with_collisions.csv') 
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

#if __name__ == "__main__":
    #app.run_server(debug=True, port=8060)
    #print(os.getcwd())
    #print("Current Working Directory:", os.getcwd())

if __name__=="__main__":
  app.run(host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']))

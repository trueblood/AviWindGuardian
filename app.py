# -*- coding: utf-8 -*-
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
from scripts.ai_data_dispatcher import AIDataDispatcher
from scripts.randomforest import RandomForest
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME],
)

#  make dataframe from  spreadsheet:
df = pd.read_csv("assets/historic.csv")

MAX_YR = df.Year.max()
MIN_YR = df.Year.min()
START_YR = 2007

# since data is as of year end, need to add start year
df = (
    df._append({"Year": MIN_YR - 1}, ignore_index=True)
    .sort_values("Year", ignore_index=True)
    .fillna(0)
)

COLORS = {
    "cash": "#3cb521",
    "bonds": "#fd7e14",
    "stocks": "#446e9b",
    "inflation": "#cd0200",
    "background": "whitesmoke",
}

"""
==========================================================================
Markdown Text
"""

datasource_text = dcc.Markdown(
    """
    [Data source:](http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html)
    Historical Returns on Stocks, Bonds and Bills from NYU Stern School of
    Business
    """
)

asset_allocation_text = dcc.Markdown(
    """
> **Asset allocation** is one of the main factors that drive portfolio risk and returns.   Play with the app and see for yourself!

> Change the allocation to cash, bonds and stocks on the sliders and see how your portfolio performs over time in the graph.
  Try entering different time periods and dollar amounts too.
"""
)

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

cagr_text = dcc.Markdown(
    """
    (CAGR) is the compound annual growth rate.  It measures the rate of return for an investment over a period of time, 
    such as 5 or 10 years. The CAGR is also called a "smoothed" rate of return because it measures the growth of
     an investment as if it had grown at a steady rate on an annually compounded basis.
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
Tables
"""

total_returns_table = dash_table.DataTable(
    id="total_returns",
    columns=[{"id": "Year", "name": "Year", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format": {"specifier": "$,.0f"}}
        for col in ["Cash", "Bonds", "Stocks", "Total"]
    ],
    page_size=15,
    style_table={"overflowX": "scroll"},
)

annual_returns_pct_table = dash_table.DataTable(
    id="annual_returns_pct",
    columns=(
        [{"id": "Year", "name": "Year", "type": "text"}]
        + [
            {"id": col, "name": col, "type": "numeric", "format": {"specifier": ".1%"}}
            for col in df.columns[1:]
        ]
    ),
    data=df.to_dict("records"),
    sort_action="native",
    page_size=15,
    style_table={"overflowX": "scroll"},
)


def make_summary_table(dff):
    """Make html table to show cagr and  best and worst periods"""

    table_class = "h5 text-body text-nowrap"
    cash = html.Span(
        [html.I(className="fa fa-money-bill-alt"), " Cash"], className=table_class
    )
    bonds = html.Span(
        [html.I(className="fa fa-handshake"), " Bonds"], className=table_class
    )
    stocks = html.Span(
        [html.I(className="fa fa-industry"), " Stocks"], className=table_class
    )
    inflation = html.Span(
        [html.I(className="fa fa-ambulance"), " Inflation"], className=table_class
    )

    start_yr = dff["Year"].iat[0]
    end_yr = dff["Year"].iat[-1]

    df_table = pd.DataFrame(
        {
            "": [cash, bonds, stocks, inflation],
            f"Rate of Return (CAGR) from {start_yr} to {end_yr}": [
                cagr(dff["all_cash"]),
                cagr(dff["all_bonds"]),
                cagr(dff["all_stocks"]),
                cagr(dff["inflation_only"]),
            ],
            f"Worst 1 Year Return": [
                worst(dff, "3-mon T.Bill"),
                worst(dff, "10yr T.Bond"),
                worst(dff, "S&P 500"),
                "",
            ],
        }
    )
    return dbc.Table.from_dataframe(df_table, bordered=True, hover=True)

"""
==========================================================================
Map
"""

def make_map():
    """
    Function to create a map with specific edit controls.
    """
    return dl.Map(center=[56, 10], zoom=4, children=[
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

def make_pie(slider_input, title):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Cash", "Bonds", "Stocks"],
                values=slider_input,
                textinfo="label+percent",
                textposition="inside",
                marker={"colors": [COLORS["cash"], COLORS["bonds"], COLORS["stocks"]]},
                sort=False,
                hoverinfo="none",
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(b=25, t=75, l=35, r=25),
        height=325,
        paper_bgcolor=COLORS["background"],
    )
    return fig


def make_line_chart(dff):
    start = dff.loc[1, "Year"]
    yrs = dff["Year"].size - 1
    dtick = 1 if yrs < 16 else 2 if yrs in range(16, 30) else 5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_cash"],
            name="All Cash",
            marker_color=COLORS["cash"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_bonds"],
            name="All Bonds (10yr T.Bonds)",
            marker_color=COLORS["bonds"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_stocks"],
            name="All Stocks (S&P500)",
            marker_color=COLORS["stocks"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["Total"],
            name="My Portfolio",
            marker_color="black",
            line=dict(width=6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["inflation_only"],
            name="Inflation",
            visible=True,
            marker_color=COLORS["inflation"],
        )
    )
    fig.update_layout(
        title=f"Returns for {yrs} years starting {start}",
        template="none",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(tickprefix="$", fixedrange=True),
        xaxis=dict(title="Year Ended", fixedrange=True, dtick=dtick),
    )
    return fig


"""
==========================================================================
Make Tabs
"""

# =======Play tab components

cords_card = dbc.Card(
    [
        dbc.CardBody(
            html.Div(id="coords-display-container", style={'overflowY': 'scroll', 'height': '300px'}),
        ),
        html.Button("Draw marker", id="draw_marker", className="mt-2"),
        html.Button("Remove -> Clear all", id="clear_all", className="mt-2"),
        html.Button("Submit", id="submit_cords", className="mt-2"),
    ],
    className="mt-4",
)

def make_coords_table(coords):
    """Generate an HTML table based on coordinates."""
    df_table = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
    return dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)




asset_allocation_card = dbc.Card(asset_allocation_text, className="mt-2")

slider_card = dbc.Card(
    [
        html.H4("First set cash allocation %:", className="card-title"),
        dcc.Slider(
            id="cash",
            marks={i: f"{i}%" for i in range(0, 101, 10)},
            min=0,
            max=100,
            step=5,
            value=10,
            included=False,
        ),
        html.H4(
            "Then set stock allocation % ",
            className="card-title mt-3",
        ),
        html.Div("(The rest will be bonds)", className="card-title"),
        dcc.Slider(
            id="stock_bond",
            marks={i: f"{i}%" for i in range(0, 91, 10)},
            min=0,
            max=90,
            step=5,
            value=50,
            included=False,
        ),
    ],
    body=True,
    className="mt-4",
)


time_period_data = [
    {
        "label": f"2007-2008: Great Financial Crisis to {MAX_YR}",
        "start_yr": 2007,
        "planning_time": MAX_YR - START_YR + 1,
    },
    {
        "label": "1999-2010: The decade including 2000 Dotcom Bubble peak",
        "start_yr": 1999,
        "planning_time": 10,
    },
    {
        "label": "1969-1979:  The 1970s Energy Crisis",
        "start_yr": 1970,
        "planning_time": 10,
    },
    {
        "label": "1929-1948:  The 20 years following the start of the Great Depression",
        "start_yr": 1929,
        "planning_time": 20,
    },
    {
        "label": f"{MIN_YR}-{MAX_YR}",
        "start_yr": "1928",
        "planning_time": MAX_YR - MIN_YR + 1,
    },
]


time_period_card = dbc.Card(
    [
        html.H4(
            "Or select a time period:",
            className="card-title",
        ),
        dbc.RadioItems(
            id="time_period",
            options=[
                {"label": period["label"], "value": i}
                for i, period in enumerate(time_period_data)
            ],
            value=0,
            labelClassName="mb-2",
        ),
    ],
    body=True,
    className="mt-4",
)

# ======= InputGroup components

start_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Amount $"),
        dbc.Input(
            id="starting_amount",
            placeholder="Min $10",
            type="number",
            min=10,
            value=10000,
        ),
    ],
    className="mb-3",
)
start_year = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Year"),
        dbc.Input(
            id="start_yr",
            placeholder=f"min {MIN_YR}   max {MAX_YR}",
            type="number",
            min=MIN_YR,
            max=MAX_YR,
            value=START_YR,
        ),
    ],
    className="mb-3",
)
number_of_years = dbc.InputGroup(
    [
        dbc.InputGroupText("Number of Years:"),
        dbc.Input(
            id="planning_time",
            placeholder="# yrs",
            type="number",
            min=1,
            value=MAX_YR - START_YR + 1,
        ),
    ],
    className="mb-3",
)
end_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Ending Amount"),
        dbc.Input(id="ending_amount", disabled=True, className="text-black"),
    ],
    className="mb-3",
)
rate_of_return = dbc.InputGroup(
    [
        dbc.InputGroupText(
            "Rate of Return(CAGR)",
            id="tooltip_target",
            className="text-decoration-underline",
        ),
        dbc.Input(id="cagr", disabled=True, className="text-black"),
        dbc.Tooltip(cagr_text, target="tooltip_target"),
    ],
    className="mb-3",
)

input_groups = html.Div(
    [start_amount, start_year, number_of_years, end_amount, rate_of_return],
    className="mt-4 p-4",
)


# =====  Results Tab components

results_card = dbc.Card(
    [
        dbc.CardHeader("My Portfolio Returns - Rebalanced Annually"),
        html.Div(total_returns_table),
    ],
    className="mt-4",
)


data_source_card = dbc.Card(
    [
        dbc.CardHeader("Source Data: Annual Total Returns"),
        html.Div(annual_returns_pct_table),
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
            [cords_card, slider_card, input_groups, time_period_card],
            tab_id="tab-2",
            label="Play",
            className="pb-4",
        ),
        dbc.Tab([results_card, data_source_card], tab_id="tab-3", label="Results"),
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)


"""
==========================================================================
Helper functions to calculate investment results, cagr and worst periods
"""


def backtest(stocks, cash, start_bal, nper, start_yr):
    """calculates the investment returns for user selected asset allocation,
    rebalanced annually and returns a dataframe
    """

    end_yr = start_yr + nper - 1
    cash_allocation = cash / 100
    stocks_allocation = stocks / 100
    bonds_allocation = (100 - stocks - cash) / 100

    # Select time period - since data is for year end, include year prior
    # for start ie year[0]
    dff = df[(df.Year >= start_yr - 1) & (df.Year <= end_yr)].set_index(
        "Year", drop=False
    )
    dff["Year"] = dff["Year"].astype(int)

    # add columns for My Portfolio returns
    dff["Cash"] = cash_allocation * start_bal
    dff["Bonds"] = bonds_allocation * start_bal
    dff["Stocks"] = stocks_allocation * start_bal
    dff["Total"] = start_bal
    dff["Rebalance"] = True

    # calculate My Portfolio returns
    for yr in dff.Year + 1:
        if yr <= end_yr:
            # Rebalance at the beginning of the period by reallocating
            # last period's total ending balance
            if dff.loc[yr, "Rebalance"]:
                dff.loc[yr, "Cash"] = dff.loc[yr - 1, "Total"] * cash_allocation
                dff.loc[yr, "Stocks"] = dff.loc[yr - 1, "Total"] * stocks_allocation
                dff.loc[yr, "Bonds"] = dff.loc[yr - 1, "Total"] * bonds_allocation

            # calculate this period's  returns
            dff.loc[yr, "Cash"] = dff.loc[yr, "Cash"] * (
                1 + dff.loc[yr, "3-mon T.Bill"]
            )
            dff.loc[yr, "Stocks"] = dff.loc[yr, "Stocks"] * (1 + dff.loc[yr, "S&P 500"])
            dff.loc[yr, "Bonds"] = dff.loc[yr, "Bonds"] * (
                1 + dff.loc[yr, "10yr T.Bond"]
            )
            dff.loc[yr, "Total"] = dff.loc[yr, ["Cash", "Bonds", "Stocks"]].sum()

    dff = dff.reset_index(drop=True)
    columns = ["Cash", "Stocks", "Bonds", "Total"]
    dff[columns] = dff[columns].round(0)

    # create columns for when portfolio is all cash, all bonds or  all stocks,
    #   include inflation too
    #
    # create new df that starts in yr 1 rather than yr 0
    dff1 = (dff[(dff.Year >= start_yr) & (dff.Year <= end_yr)]).copy()
    #
    # calculate the returns in new df:
    columns = ["all_cash", "all_bonds", "all_stocks", "inflation_only"]
    annual_returns = ["3-mon T.Bill", "10yr T.Bond", "S&P 500", "Inflation"]
    for col, return_pct in zip(columns, annual_returns):
        dff1[col] = round(start_bal * (1 + (1 + dff1[return_pct]).cumprod() - 1), 0)
    #
    # select columns in the new df to merge with original
    dff1 = dff1[["Year"] + columns]
    dff = dff.merge(dff1, how="left")
    # fill in the starting balance for year[0]
    dff.loc[0, columns] = start_bal
    return dff


def cagr(dff):
    """calculate Compound Annual Growth Rate for a series and returns a formated string"""

    start_bal = dff.iat[0]
    end_bal = dff.iat[-1]
    planning_time = len(dff) - 1
    cagr_result = ((end_bal / start_bal) ** (1 / planning_time)) - 1
    return f"{cagr_result:.1%}"


def worst(dff, asset):
    """calculate worst returns for asset in selected period returns formated string"""

    worst_yr_loss = min(dff[asset])
    worst_yr = dff.loc[dff[asset] == worst_yr_loss, "Year"].iloc[0]
    return f"{worst_yr_loss:.1%} in {worst_yr}"


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
                        dcc.Graph(id="allocation_pie_chart", className="mb-2"),
                        dcc.Graph(id="returns_chart", className="pb-4"),
                        html.Hr(),
                        html.Div(id="summary_table"),
                        html.H6(datasource_text, className="my-2"),
                        html.Div(id="prediction-output"),
                        html.Div(id='coords-json', style={'display': 'none'})
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


@app.callback(
    Output("allocation_pie_chart", "figure"),
    Input("stock_bond", "value"),
    Input("cash", "value"),
)
def update_pie(stocks, cash):
    bonds = 100 - stocks - cash
    slider_input = [cash, bonds, stocks]

    if stocks >= 70:
        investment_style = "Aggressive"
    elif stocks <= 30:
        investment_style = "Conservative"
    else:
        investment_style = "Moderate"
    figure = make_pie(slider_input, investment_style + " Asset Allocation")
    return figure


@app.callback(
    Output("stock_bond", "max"),
    Output("stock_bond", "marks"),
    Output("stock_bond", "value"),
    Input("cash", "value"),
    State("stock_bond", "value"),
)
def update_stock_slider(cash, initial_stock_value):
    max_slider = 100 - int(cash)
    stocks = min(max_slider, initial_stock_value)

    # formats the slider scale
    if max_slider > 50:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 10)}
    elif max_slider <= 15:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 1)}
    else:
        marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 5)}
    return max_slider, marks_slider, stocks


@app.callback(
    Output("planning_time", "value"),
    Output("start_yr", "value"),
    Output("time_period", "value"),
    Input("planning_time", "value"),
    Input("start_yr", "value"),
    Input("time_period", "value"),
)
def update_time_period(planning_time, start_yr, period_number):
    """syncs inputs and selected time periods"""
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if input_id == "time_period":
        planning_time = time_period_data[period_number]["planning_time"]
        start_yr = time_period_data[period_number]["start_yr"]

    if input_id in ["planning_time", "start_yr"]:
        period_number = None

    return planning_time, start_yr, period_number


@app.callback(
    Output("total_returns", "data"),
    Output("returns_chart", "figure"),
    Output("summary_table", "children"),
    Output("ending_amount", "value"),
    Output("cagr", "value"),
    Input("stock_bond", "value"),
    Input("cash", "value"),
    Input("starting_amount", "value"),
    Input("planning_time", "value"),
    Input("start_yr", "value"),
)
def update_totals(stocks, cash, start_bal, planning_time, start_yr):
    # set defaults for invalid inputs
    start_bal = 10 if start_bal is None else start_bal
    planning_time = 1 if planning_time is None else planning_time
    start_yr = MIN_YR if start_yr is None else int(start_yr)

    # calculate valid planning time start yr
    max_time = MAX_YR + 1 - start_yr
    planning_time = min(max_time, planning_time)
    if start_yr + planning_time > MAX_YR:
        start_yr = min(df.iloc[-planning_time, 0], MAX_YR)  # 0 is Year column

    # create investment returns dataframe
    dff = backtest(stocks, cash, start_bal, planning_time, start_yr)

    # create data for DataTable
    data = dff.to_dict("records")

    # create the line chart
    fig = make_line_chart(dff)

    summary_table = make_summary_table(dff)

    # format ending balance
    ending_amount = f"${dff['Total'].iloc[-1]:0,.0f}"

    # calcluate cagr
    ending_cagr = cagr(dff["Total"])

    return data, fig, summary_table, ending_amount, ending_cagr

# Trigger mode (draw marker).
@app.callback(Output("edit_control", "drawToolbar"), Input("draw_marker", "n_clicks"))
def trigger_mode(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dict(mode="marker", n_clicks=n_clicks)

# Trigger mode (edit) + action (remove all).
@app.callback(Output("edit_control", "editToolbar"), Input("clear_all", "n_clicks"))
def trigger_action(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dict(mode="remove", action="clear all", n_clicks=n_clicks)

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

# Helper function to format predictions for display
def format_predictions(predictions_df):
    # Formatting the DataFrame as an HTML table or list
    return html.Ul([html.Li(f"Longitude: {row['xlong']}, Latitude: {row['ylat']}, Prediction: {row['prediction']}")
                    for index, row in predictions_df.iterrows()])


# Trigger mode (submit)
@app.callback(
    [Output("prediction-output", "children")],
    [Input("submit_cords", "n_clicks")],
    [State("edit_control", "geojson"),
     State("coords-json", "children")]
)
def trigger_action_and_predict(n_clicks, geojson, json_coords):
    if n_clicks is None or not geojson:
        raise PreventUpdate
    
    # Convert JSON back to DataFrame
    df_coords = pd.read_json(json_coords, orient='split')
    #print(df_coords.head())
    #df_coords = convert_geojson_to_dataframe(geojson)
    
    # Assuming geojson to DataFrame conversion is done here
    #df = convert_geojson_to_dataframe(geojson)
    #print(df_coords.head())
    #trainModel()
    # Initialize your RandomForest and model
    randomForest = RandomForest()
    model = randomForest.load_model('random_forest_model.joblib')
    #trainModel()
    #test_data = pd.DataFrame({
    #        'xlong': [-99.787033, -99.725624, -99.769722, -99.80706, -99.758476, -99.772133, -99.740372, -99.810005, -99.762489, -99.761673, -99.776581, -99.724426, -99.714981, -99.70752, -99.760849, -99.745461, -99.733009, -99.750793, -99.814697, -99.742203, -99.737236, -99.741096, -99.781944, -99.796494, -99.733086, -99.775742, -99.724144, -99.786308, -99.737022, -99.776512, -99.741119, -99.735718, -99.762627, -99.74147, -99.804855, -99.792625, -99.788475, -99.799377, -99.79464, -99.756958, -99.790665, -99.720284, -99.714928, -99.782089, -99.821129, -99.744316, -99.731606, -99.778603, -99.782372, -99.826469, -99.751343, -99.752548, -99.775963, -99.764076, -99.812325, -99.809586, -99.752739, -99.771027, -99.720001, -99.746208, -118.364197, -118.363762, -118.36441, -93.518082, -93.632835, -93.523651, -93.623009, -93.700424, -93.430367, -92.672089, -93.51371, -93.515892, -93.367798, -70.541801, -70.545303, -70.547798, -70.545303, -93.325691, -93.428093, -93.431992, -93.354897, -93.632095, -93.636795, -83.736298, -83.736298, -94.65139, -94.707893, -94.685692, -94.683395, -94.673988, -94.695091, -94.68869, -94.61039, -94.692795, -94.706894, -94.687691, -94.659096, -94.674889, -94.68409, -94.68869, -94.665192, -94.670395, -94.618591, -94.69899, -94.723892, -94.611893, -94.658493, -94.62249, -94.669289, -94.634491, -94.666992, -94.676788, -94.677391, -94.689392, -94.71209, -94.652596, -94.613396, -94.665092, -94.703896, -94.688995, -94.726593, -94.63839, -94.655289, -94.647896, -94.612091, -94.67009, -94.619591, -94.654495, -94.683502, -94.674194, -94.614189, -94.673492, -94.630989, -94.692696, -94.670593, -94.647491, -94.615791, -94.660995, -94.715591, -94.704094, -94.696693, -94.716591, -94.722794, -94.622093, -94.695595, -94.730293, -94.618294, -94.678894, -94.676994, -94.711792, -94.606728, -94.681793, -94.635391, -94.691689, -94.659866, -94.655891, -94.655891, -94.648689, -94.630692, -94.665695, -94.626991, -94.686043, -94.697762, -94.671776, -94.649178, -94.734169, -94.724846, -94.72422, -94.662544, -94.664101, -94.664093, -94.647606, -94.725616, -94.639961, -94.678551, -94.665192, -94.716476, -94.669571, -94.659653, -94.712288, -94.705017, -94.673912, -94.713112, -94.664482, -94.728653, -94.695923, -94.66095, -94.657524, -94.712463, -94.649925, -94.667892, -94.634834, -94.728088, -94.697464, -94.616638, -94.721962, -94.65696, -94.615784, -94.716339],
    #        'ylat': [36.501724, 36.437126, 36.444931, 36.513935, 36.444984, 36.431931, 36.489838, 36.476582, 36.454903, 36.502792, 36.485386, 36.491375, 36.490211, 36.490849, 36.429668, 36.448009, 36.489468, 36.488712, 36.48975, 36.432888, 36.498882, 36.423683, 36.433651, 36.503357, 36.451591, 36.445465, 36.421703, 36.44593, 36.435909, 36.429882, 36.50259, 36.423359, 36.515823, 36.451477, 36.476624, 36.474033, 36.476807, 36.440704, 36.438831, 36.491249, 36.44141, 36.424934, 36.424843, 36.473541, 36.517494, 36.513348, 36.44109, 36.502522, 36.484463, 36.515972, 36.446674, 36.428902, 36.458096, 36.440956, 36.513859, 36.495914, 36.504105, 36.456665, 36.489346, 36.427345, 35.077644, 35.077908, 35.077435, 42.01363, 41.882477, 42.006813, 41.88147, 41.977608, 42.028233, 41.742046, 42.019119, 42.016373, 42.49794, 41.752491, 41.754192, 41.75959, 41.757591, 42.20639, 42.146091, 42.145592, 41.904194, 42.335491, 42.335491, 41.382492, 41.384693, 41.421494, 41.470394, 41.471092, 41.484993, 41.494892, 41.457893, 41.486591, 41.441395, 41.499294, 41.440792, 41.445091, 41.466694, 41.423492, 41.443592, 41.499294, 41.452293, 41.434994, 41.466793, 41.457794, 41.463192, 41.452892, 41.433193, 41.466595, 41.452293, 41.439693, 41.482494, 41.452393, 41.463093, 41.470192, 41.468391, 41.455894, 41.467094, 41.423695, 41.499695, 41.435894, 41.442894, 41.439693, 41.466293, 41.465393, 41.478592, 41.495193, 41.451992, 41.432995, 41.433804, 41.434193, 41.441395, 41.463093, 41.481194, 41.434994, 41.423992, 41.443691, 41.452694, 41.421093, 41.441792, 41.471092, 41.499092, 41.469791, 41.443394, 41.441994, 41.445194, 41.442394, 41.441593, 41.443993, 41.473293, 41.441292, 41.441357, 41.471592, 41.449795, 41.445091, 41.457573, 41.421394, 41.457294, 41.455395, 41.439194, 41.470993, 41.481194, 40.907246, 40.927402, 40.91991, 40.947777, 40.903179, 40.906811, 40.91663, 40.91256, 40.952431, 40.931583, 40.912476, 40.925251, 40.918289, 40.951794, 40.902493, 40.913757, 40.902508, 40.925396, 40.912891, 40.916325, 40.955795, 40.929382, 40.9258, 40.905937, 40.901394, 40.906422, 40.913597, 40.923447, 40.917538, 40.919884, 40.90395, 40.916672, 40.908005, 40.913837, 40.927204, 40.905529, 40.906044, 40.92387]
    #})
    
    predictions = model.predict_with_location(df_coords)
    print(predictions)
    
    table = dbc.Table.from_dataframe(predictions, striped=True, bordered=True, hover=True)
    # Make predictions
    #predictions = ai_dispatcher.predictRandomForest(model, df)
    #predictions = randomForest.predict(test_data)
    
    # Format predictions for display
    prediction_output = format_predictions(predictions)
    
    return prediction_output

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

    mean_accuracy, std_dev = randomForest.evaluate_model_with_kfold(X, Y, n_splits=2)
    print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_dev}")
    

# Helper function to convert GeoJSON to DataFrame
def convert_geojson_to_dataframe(geojson):
    # Implement conversion logic
    return pd.DataFrame()

# Helper function to format predictions for display
def format_predictions(predictions):
    # Implement formatting logic
    return html.Div(str(predictions))  # Simple example

# Callback to display coordinates directly from the edit_control's geojson data.
'''
@app.callback(Output("coords-display-container", "children"), Input("edit_control", "geojson"))
def display_coords(geojson):
    if not geojson or "features" not in geojson:
        raise PreventUpdate
    rows = []  # List to hold row data
    point_counter, polygon_counter = 1, 1  # Initialize counters
    
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")
        
        if geom_type == "Point":
            lat, lon = coords[1], coords[0]
            label = f"Point {point_counter}"
            rows.append({"Type": label, "Latitude": lat, "Longitude": lon})
            point_counter += 1  # Increment point counter
            
        elif geom_type == "Polygon":
            vertex_counter = 1  # Initialize vertex counter for each polygon
            for coord in coords[0]:  # Assuming the first ring of coordinates for simplicity
                lat, lon = coord[1], coord[0]
                label = f"Polygon {polygon_counter} Vertex {vertex_counter}"
                rows.append({"Type": label, "Latitude": lat, "Longitude": lon})
                vertex_counter += 1  # Increment vertex counter
            polygon_counter += 1  # Increment polygon counter
            
    if not rows:
        return "No coordinates available"

    # Convert rows into a DataFrame and then into a dbc.Table
    df_coords = pd.DataFrame(rows)
    table = dbc.Table.from_dataframe(df_coords, striped=True, bordered=True, hover=True)
    return table
'''
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
            rows.append({"group": pointGroup, "#": point_counter, "Type": label,"Latitude": lat, "Longitude": lon, "Prediction": "N/A"})
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
                rows.append({"group": pointGroup, "#": polygon_counter, "Type": label, "Latitude": lat, "Longitude": lon, "Prediction": "N/A"})
            polygon_counter += 1  # Increment polygon counter after processing all vertices of a polygon
            pointGroup += 1
            
    if not rows:
        return "No coordinates available"

    # Convert rows into a DataFrame and then into a dbc.Table
    df_coords = pd.DataFrame(rows)
    table = dbc.Table.from_dataframe(df_coords, striped=True, bordered=True, hover=True)
     # Convert DataFrame to JSON for easy transmission
    json_coords = df_coords.to_json(orient='split')
    return table, json_coords


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    print(os.getcwd())
    print("Current Working Directory:", os.getcwd())
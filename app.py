from dash import dcc, Dash, html, dash_table, callback, Output, Input
import pandas as pd
import plotly.express as px
from tools import *

global_df = generate_global_df()
df = global_df[global_df['Country'] == global_df['Country'].unique()[0]]
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Population Data by Country"),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': i, 'value': i} for i in global_df['Country'].unique()],
        value=global_df['Country'][0],  # Default value
    ),
    dash_table.DataTable(data=df.to_dict('records'), page_size=6)
])
@app.callback(
    Output('df', 'children'),
    [Input('country-dropdown', 'value')]
)
def update_population_display(selected_country):
    df = global_df[global_df['Country'] ==selected_country]
    return df.to_html()

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

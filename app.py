import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tools import *
from dash.dependencies import Input, Output
import plotly.express as px
from dash import Dash, html,dcc,dash_table
from plotly.tools import mpl_to_plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc

# Read the CSV file
df_full = generate_global_df()

df_hotel_rooms = df_full[['Url link'] +list(set(columns_room) - set(columns_hotel))]

df_hotels = df_full.groupby(['Url link', 'Location']).first().reset_index()[['City']+list(set(columns_hotel)-set(columns_city))]
df_hotels.insert(0, 'Distance from center', df_hotels.pop('Distance from center'))
df_hotels.insert(0, 'Score', df_hotels.pop('Score'))
df_hotels.insert(0, 'Stars', df_hotels.pop('Stars'))
df_hotels.insert(0, 'Location', df_hotels.pop('Location'))
df_hotels.insert(0, 'Name', df_hotels.pop('Name'))

df_cities = df_full.groupby('City').first().reset_index()[['Country']+list(set(columns_city)-set(columns_country))]
df_cities.insert(0, 'Country', df_cities.pop('Country'))
df_cities.insert(0, 'City', df_cities.pop('City'))

df_countries = df_full.groupby('Country').first().reset_index()[columns_country]

df_cities_countries = pd.merge(df_countries,df_cities, on='Country', how='inner')
df_cities_countries.insert(0, 'Country', df_cities_countries.pop('Country'))
df_cities_countries.insert(0, 'City', df_cities_countries.pop('City'))


df_countries.insert(1, 'Population', df_countries.pop('Population'))
df_countries.insert(2, 'Yearly Change', df_countries.pop('Yearly Change'))
continents = df_countries['Continent'].unique()
df_countries['Continent_n'] = df_countries['Continent'].apply(lambda x: np.where(continents==x)[0][0])
# plot_country_bubbles(df_countries, v_name_color='Continent_n', title='The countries use in the study', cmap_color='crest_r')
df_countries.drop(columns=['Continent_n'], inplace=True)


df = pd.merge(df_cities, df_countries, on='Country', how='inner')
df = pd.merge(df, df_hotels, on='City', how='inner')


# ! comparer les modeles (les metriques de chaque modele avec des tests msh bs accuracy w kholset)

# Generate the seaborn plot
sns_plot = sns.displot(df_full, x="Score", kind="hist", hue="Continent", discrete=True, multiple="stack")
plt.title("Distrubution of score")
plt.ylabel("Occurance")
plt.xlabel("Score of hotels")

# Convert the seaborn plot to a plotly figure
plotly_fig = tls.mpl_to_plotly(sns_plot.fig)
# Adjust the size of the figure
plotly_fig.layout.width = 800  # adjust as needed
plotly_fig.layout.height = 600 

intro_text = (
    "The hotels industry is a dynamic and ever-evolving sector, shaping the global travel experience. "
    "From bustling cityscapes to serene countryside retreats, hotels cater to diverse needs and preferences. "
    "In this interactive exploration, we'll delve into the fascinating details that make hotels tick, "
    "examining both the internal features that enhance guest comfort and the external factors like location "
    "that influence their appeal. Let's embark on a data-driven journey to unlock insights about the world of hotels!"
)

plt.clf()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

primary_color = "#4B0082"
secondary_color = "#f8f9fa"
text_color = "#333"

app.layout = dbc.Container([
    # Intro Row with Text and Separator
    dbc.Row([
        dbc.Col(
            [
                html.H1(children="Exploring the Hotels Industry", className='text-center text-primary fw-bold mb-3'),
                html.P(intro_text, className='lead text-color')
            ],
            width=12,
            style={'padding': '20px 0px'}
        )
    ]),

    dbc.Row([
         html.H2("1.Hotels Features", className='text-center text-secondary fw-bold mb-3'),

    ]),

    dbc.Row([
        dbc.Col(
            html.Hr(style={'borderWidth': '1px', 'borderColor': '#ddd'}),
            width=12
        )
    ]),

    # Filter and Chart Row
    dbc.Row([
        dbc.Col(
            [
                # dbc.Label('Chart Type', className='h5 text-primary mb-3'),
                dcc.RadioItems(
                    id='chart-type',
                    options=[
                        {'label': 'Hotel Counts', 'value': 'hotels'},
                        {'label': 'Room Counts', 'value': 'rooms'}
                    ],
                    value='hotels',  # default value
                    className='form-check-inline mb-3'
                ),
                dcc.Graph(id='pie-chart', config={'displayModeBar': False})
            ],
            md=6,
            # className='border-end'
        ),
        dbc.Col(
            [
                # dbc.Label('Chart Type', className='h7 text-primary mb-3'),
                dcc.RadioItems(
                    id='chart-type2',
                    options=[
                        {'label': 'Breakfast', 'value': 'breakfast'},
                        {'label': 'Cancellation', 'value': 'cancellation'}
                    ],
                    value='breakfast',  # default value
                    className='form-check-inline mb-3'
                ),
                dcc.Graph(id='pie-chart2', config={'displayModeBar': False})
            ],
            md=6
        ),
        html.Div(style={'border-left': '5px solid #ddd', 'height': '100%', 'margin': '0 10px'})
    ]),

    # Separator
    dbc.Row([
        dbc.Col(
            html.Hr(style={'borderWidth': '2px', 'borderColor': primary_color,}),
            width=12
        )
    ],style={'margin-bottom': '20px'}),

    # Third Filter and Chart Row
    dbc.Row([
        dbc.Col(
            [
                # dbc.Label('Chart Type', className='h5 text-primary mb-3'),
                dcc.RadioItems(
                    id='chart-type-2',
                    options=[
                        {'label': 'Score', 'value': 'Score'},
                        {'label': 'Stars', 'value': 'Stars'},
                        {'label': 'Room Capacity', 'value': 'Guests nb'}
                    ],
                    value='Score',  # default value
                    className='form-check-inline mb-3'
                ),
                dcc.Graph(id='graph')
            ],
            md=12
        ),

    ]),

    # Separator
    dbc.Row([
        dbc.Col(
            html.Hr(style={'borderWidth': '2px', 'borderColor': primary_color}),
            width=12
        )
    ],style={'margin-bottom': '50px', 'margin-top': '20px'}),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='continent-dropdown', placeholder='Select Continent',value=df_full['Continent'].unique()[0]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='rating-distribution')
                ])
            ])
        ])


    ])
])


@app.callback(
    Output('pie-chart', 'figure'),
    [Input('chart-type', 'value')]
)

def update_pie_chart(chart_type):
    if chart_type == 'hotels':
        hotel_counts = df['Continent'].value_counts()
        total_hotels = hotel_counts.sum()
        data = [go.Pie(labels=hotel_counts.index, values=hotel_counts, textinfo='label+percent', hole=0.3)]
        title = f'Number of Hotels used in the study: {total_hotels} hotels'
    else:
        df_merged = pd.merge(df, df_hotel_rooms, on='Url link', how='inner')
        room_counts = df_merged['Continent'].value_counts()
        total_rooms = room_counts.sum()
        data = [go.Pie(labels=room_counts.index, values=room_counts, textinfo='label+percent', hole=0.3)]
        title = f'Number of Rooms used in the study: {total_rooms} rooms'
    
    layout = go.Layout(title=title, title_font=dict(color="black"), font=dict(color="black"))
    return {'data': data, 'layout': layout}

@app.callback(
    Output('graph', 'figure'),
    [Input('chart-type-2', 'value')]
)

def update_graph(selected_value):
    if selected_value == 'Score':
        fig = px.histogram(df_full, x="Score", color="Continent", nbins=10, histnorm='percent')
        fig.update_layout(title_text='Distribution of scores', xaxis_title='Score of hotels', yaxis_title='Occurrence')
    elif selected_value == 'Stars':
        fig = px.histogram(df_full, x="Stars", color="Continent", nbins=10, histnorm='percent')
        fig.update_layout(title_text='Distribution of stars', xaxis_title='Stars of hotels', yaxis_title='Occurrence')
    elif selected_value == 'Guests nb':
        fig = px.histogram(df_full, x="Guests nb", color="Continent", nbins=10, histnorm='percent')
        fig.update_layout(title_text='Room capacity', xaxis_title='Room capacity (# of people)', yaxis_title='Occurrence')

    fig.update_layout(  
        legend=dict(
            title=dict(text="Continent"),
            orientation="h",
            yanchor="bottom",
            y=0.5,
            xanchor="right",
            x=0.5,
              # Adjust the y value similarly if needed
        )
    )

    return fig


@app.callback(
    Output('pie-chart2', 'figure'),
    [Input('chart-type2', 'value')]
)
def update_pie_chart2(chart_type):
    if chart_type == 'breakfast':
        breakfast_counts = df_full['Breakfast'].value_counts()
        breakfast_df = pd.DataFrame({
            'Type of Breakfast': ['Offered', 'No Breakfast', 'Paid'],
            'Percentage of Hotels (%)': breakfast_counts[[0, -1, 1]].values
        })
        fig = go.Figure(data=[go.Pie(labels=breakfast_df['Type of Breakfast'], values=breakfast_df['Percentage of Hotels (%)'], hole=0.3)])
        fig.update_layout(title_text="Breakfast services")
    else:
        counts = df_full["Cancellation"].value_counts()
        counts_df = pd.DataFrame({
            'Cancellation Policy': ['Free cancellation' if index == 1 else 'Non-refundable' for index in counts.index],
            'Count': counts.values
        })
        fig = go.Figure(data=[go.Pie(labels=counts_df['Cancellation Policy'], values=counts_df['Count'], hole=0.3)])
        fig.update_layout(title_text="Cancellation policy")
    
    return fig


@app.callback(
    Output('rating-distribution', 'figure'),
    [Input('continent-dropdown', 'value')],

)
def update_rating_distribution(selected_continent):
    fig = go.Figure()

    for rating_col in ['Staff_rating', 'Facilities_rating', 'Cleanliness_rating',
                       'Comfort_rating', 'Value_for_money_rating', 'Location_rating']:
        filtered_df = df_full[df_full['Continent'] == selected_continent]
        fig.add_trace(go.Histogram(x=filtered_df[rating_col], name=rating_col))

    fig.update_layout(title=f'Distribution of Ratings in {selected_continent}',
                      xaxis_title='Rating',
                      yaxis_title='Frequency')
    return fig


@app.callback(
    Output('continent-dropdown', 'options'),
    [Input('chart-type', 'value')]
)
def update_continent_dropdown(chart_type):
    if chart_type == 'hotels':
        return [{'label': continent, 'value': continent} for continent in df_full['Continent'].unique()]
    else:
        return []



if __name__ == '__main__':
    app.run_server(debug=True)


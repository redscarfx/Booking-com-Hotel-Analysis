import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tools import *
import plotly.express as px
from dash import Dash, html,dcc,dash_table
from plotly.tools import mpl_to_plotly

# Read the CSV file
df_full = generate_global_df()
df_hotel_rooms = df_full[['Url link', 'Continent'] +list(set(columns_room) - set(columns_hotel))]
df_hotels = df_full.groupby(['Url link', 'Location']).first().reset_index()[['City','Continent']+list(set(columns_hotel)-set(columns_city))]
df_hotels.insert(0, 'Distance from center', df_hotels.pop('Distance from center'))
df_hotels.insert(0, 'Score', df_hotels.pop('Score'))
df_hotels.insert(0, 'Stars', df_hotels.pop('Stars'))
df_hotels.insert(0, 'Location', df_hotels.pop('Location'))
df_hotels.insert(0, 'Name', df_hotels.pop('Name'))
df_cities = df_full.groupby('City').first().reset_index()[['Country']+list(set(columns_city)-set(columns_country))]
df_countries = df_full.groupby('Country').first().reset_index()[columns_country]
df_cities_countries = pd.merge(df_countries,df_cities, on='Country', how='inner')

app = Dash(__name__)

# Create the plot 1
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig = px.histogram(df_full, x="Score", nbins=27, histnorm='probability density', color="Continent")
    fig.update_layout(
        title_text='Distrubution of score', # title of plot
        xaxis_title_text='Score of hotels', # xaxis label
        yaxis_title_text='Occurance', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )

# Create seaborn plot 2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sns.histplot(df_full, x="Guests nb",discrete=True)
    plt.title("Histogram of room capacity")
    plt.ylabel("Occurences")
    plt.xlabel("Room capacity (# of people)")
    plt.xticks(range(0,10))
    plt.xlim([0,11])
    plt.tight_layout()
    plt.savefig("temp_plot.png")
# Convert matplotlib figure to plotly figure
mpl_fig = plt.gcf()
plotly_fig = mpl_to_plotly(mpl_fig)


# Create seaborn airplot 3
vars=["Score","Facilities_rating","Cleanliness_rating","Comfort_rating","Value_for_money_rating","Staff_rating","Location_rating"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pairplot = sns.pairplot(df_hotels, vars=vars, height=2, hue="Continent")
    plt.title('Scatter plot of ratings')
pairplot.savefig("pairplot.png")

mpl_fig_pairplot = plt.gcf()
plotly_fig_pairplot = mpl_to_plotly(mpl_fig_pairplot)





# Clear the current figure
plt.clf()

app.layout = html.Div([
    html.Div(children='Helloooo World'),
    dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_full.columns],
        data=df_full.head().to_dict('records'),
    ),
    dcc.Graph(figure=fig),
    dcc.Graph(figure=plotly_fig),
    dcc.Graph(figure=plotly_fig_pairplot)
])

if __name__ == '__main__':
    app.run_server(debug=True)
import pandas as pd
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import matplotlib.patches as mpatches
plt.style.use('ggplot')

cmap_white = mcolors.LinearSegmentedColormap.from_list("white_cmap", ["white", "white"])

files_names = glob.glob('data/hotels_*.csv')
files_links_names = glob.glob('links/hotels_*.csv')

columns_country = ['Country', 'Region', 'Continent',
        'Population',
    'Yearly Change', 'net_change', 'density', 'land_area', 'migrants',
    'fert_rate', 'med_age', 'urban_pop', 'world_share', 'Tourism',
    'Tourism_year']

columns_city = ['City'] + columns_country+ ['Price per Square Meter to Buy Apartment Outside of Centre',
    'Price per Square Meter to Buy Apartment in City Centre',
    'International Primary School, Yearly for 1 Child',
    'Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child',
    '1 Pair of Jeans (Levis 501 Or Similar)',
    '1 Pair of Men Leather Business Shoes',
    '1 Pair of Nike Running Shoes (Mid-Range)',
    '1 Summer Dress in a Chain Store (Zara, H&M, ...)', 'Apples (1kg)',
    'Banana (1kg)', 'Beef Round (1kg) (or Equivalent Back Leg Red Meat)',
    'Bottle of Wine (Mid-Range)', 'Chicken Fillets (1kg)',
    'Cigarettes 20 Pack (Marlboro)', 'Domestic Beer (0.5 liter bottle)',
    'Eggs (regular) (12)', 'Imported Beer (0.33 liter bottle)_x',
    'Lettuce (1 head)', 'Loaf of Fresh White Bread (500g)',
    'Local Cheese (1kg)', 'Milk (regular), (1 liter)', 'Onion (1kg)',
    'Oranges (1kg)', 'Potato (1kg)', 'Rice (white), (1kg)', 'Tomato (1kg)',
    'Water (1.5 liter bottle)', 'Apartment (1 bedroom) Outside of Centre',
    'Apartment (1 bedroom) in City Centre',
    'Apartment (3 bedrooms) Outside of Centre',
    'Apartment (3 bedrooms) in City Centre', 'Cappuccino (regular)',
    'Coke/Pepsi (0.33 liter bottle)', 'Domestic Beer (0.5 liter draught)',
    'Imported Beer (0.33 liter bottle)_y',
    'McMeal at McDonalds (or Equivalent Combo Meal)',
    'Meal for 2 People, Mid-range Restaurant, Three-course',
    'Meal, Inexpensive Restaurant', 'Water (0.33 liter bottle)',
    'Average Monthly Net Salary (After Tax)',
    'Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate',
    'Cinema, International Release, 1 Seat',
    'Fitness Club, Monthly Fee for 1 Adult',
    'Tennis Court Rent (1 Hour on Weekend)', 'Gasoline (1 liter)',
    'Monthly Pass (Regular Price)', 'One-way Ticket (Local Transport)',
    'Taxi 1hour Waiting (Normal Tariff)', 'Taxi 1km (Normal Tariff)',
    'Taxi Start (Normal Tariff)',
    'Toyota Corolla Sedan 1.6l 97kW Comfort (Or Equivalent New Car)',
    'Volkswagen Golf 1.4 90 KW Trendline (Or Equivalent New Car)',
    'Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment',
    'Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)',
    'Mobile Phone Monthly Plan with Calls and 10GB+ Data']

columns_hotel  = ['Name',  'Location', 'Score',
    'NbReviews', 'Staff_rating', 'Facilities_rating', 'Cleanliness_rating',
    'Comfort_rating', 'Value_for_money_rating', 'Location_rating','Distance from center', 'Stars','Url link'] + columns_city

columns_room = columns_hotel + [ 'Room Type', 'Guests nb', 'Price',  'Breakfast', 'Cancellation']

def hotel_link(file_name):
    parts = file_name.split('/')
    directory = parts[0] if len(parts) > 1 else None
    file_name = parts[-1]
    parts = file_name.split('_')
    region_name = parts[1]
    region_number = int(parts[2])
    continent_name = parts[3]
    continent_number = int(parts[4].split('.')[0]) 
    ext = (parts[4].split('.')[1]) 
    return directory, region_name, region_number, continent_name, continent_number, ext

def extract_region_name(file_name):
    parts = file_name.split('/')
    file_name = parts[-1]
    parts = file_name.split('_')
    region_name = parts[1].split('.')[0]  # Split at the '.' and take the first part
    return region_name

def links_and_data_files_associate():
    global files_names, link_file_names
    """ if except_columns is not None:
        columns = [col for col in columns if col not in except_columns] """
    files_dict = {extract_region_name(name): name for name in files_names}
    files_dict = dict(sorted(files_dict.items()))
    links_dict = {hotel_link(link)[3]: link for link in files_links_names}
    links_dict = dict(sorted(links_dict.items()))
    return [(files_dict[region], links_dict[region]) for region in files_dict if region in links_dict]
    
def generate_global_df(columns=None, except_columns=None):
    global files_names, link_file_names
    """ if except_columns is not None:
        columns = [col for col in columns if col not in except_columns] """
    dfs = []
    paired_files = links_and_data_files_associate()
    for  link_file_name, file_name in list(paired_files):
        df1 = pd.read_csv(file_name)
        df2 = pd.read_csv(link_file_name)
        df = pd.merge(df1, df2, on='Url link')
        #df.drop(columns=['Url link'], inplace=True)
        df1 = pd.read_csv('data/cities.csv')
        df = pd.merge(df, df1, on=['City', 'Country','Region', 'Continent'])
        df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)
        df.drop(columns=['Rating'], inplace=True)
        dfs.append(df)
    df= pd.concat(dfs, ignore_index=True)
    #df =  df[columns]
    df = df.drop_duplicates()
    return df

def plot_country_bubbles(df, v_name_bubble=None, v_name_color=None, title=None, describe=False, cmap_color='crest'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_bubble, var_color = None, None
        
        if v_name_bubble is None or v_name_color is None:
            if df.shape[1] >= 3:
                if v_name_bubble is None:
                    var_bubble = df.columns[1]
                if v_name_color is None:
                    var_color = df.columns[2]
                if var_bubble is None:
                    var_bubble = v_name_bubble
                if var_color is None:
                    var_color = v_name_color
            elif df.shape[1] == 2:
                if v_name_bubble is None and v_name_color is not None:
                    var_color = df.columns[1]
                elif v_name_bubble is not None and v_name_color is None:
                    var_bubble = df.columns[1]
                else:
                    raise ValueError('The dataframe must have at least 3 columns')
            else:
                raise ValueError('The dataframe must have at least 3 columns')
        else:
            var_bubble, var_color = v_name_bubble, v_name_color
        if describe:
            display(df.describe().loc[['mean', 'std']])
        
        fig, ax = plt.subplots(1, 1, figsize=(60, 40))
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.boundary.plot(ax=ax, color='black', linewidth=0.8)
        world['name'] = world['name'].replace('United States of America', 'United States')
        world = world.merge(df, how='right', left_on='name', right_on='Country')
        
        if v_name_color is not None:
            world.plot(column=var_color, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, cmap=cmap_color)
        #else:
            #TODO bug
            #world.plot(ax=ax, color='none', edgecolor='black', linewidth=1.0) 


        if v_name_bubble is not None:
            for idx, row in world.iterrows():
                if not pd.isnull(row[var_color]):
                    ax.scatter(row.geometry.centroid.x, row.geometry.centroid.y, s=row[var_bubble], c='#CC5A49',alpha=1)  
        if not world[world['Country'].isnull()].empty:
            world_no_country = world[world['Country'].isnull()]
            if world_no_country.geometry.centroid.y.nunique() == 1:
                world_no_country.geometry = world_no_country.geometry.translate(yoff=1e-10)
            world_no_country.plot(ax=ax, color='none', edgecolor='black', linewidth=1.0)

        ax.set_xlabel('Longitude', fontdict={'fontsize': 40})
        ax.set_ylabel('Latitude',  fontdict={'fontsize': 40})
        if title is None:
            if var_bubble is not None:
                if var_color is not None:
                    title = f'Average {var_color} (color) & {var_bubble} (bubble) of Hotels in Each Country'
                else:
                    title = f'Average {var_bubble} of Hotels in Each Country'
            else:
                if var_color is not None:
                    title = f'Average {var_color} of Hotels in Each Country'
        plt.title(title, fontsize=40)
        plt.show()
        
def pca_analysis(df=None, except_columns=None, n_components=None, pc_x=0, pc_y=1, show_scatter_and_variance=True, circle=False):
    if df is None:
        hotel_rooms = generate_global_df()
    else:
        hotel_rooms = df
        
    hotel_rooms = hotel_rooms.select_dtypes(include=[np.number])
    hotel_rooms = hotel_rooms.apply(pd.to_numeric, errors='coerce')

    scaler = StandardScaler()
    hotel_rooms_scaled = scaler.fit_transform(hotel_rooms)

    if n_components is None:
        pca = PCA()
        n_components = len(hotel_rooms.columns)
    else:
        pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(hotel_rooms_scaled)
    principalDf = pd.DataFrame(data = principalComponents)

    if show_scatter_and_variance:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        sns.scatterplot(data=principalDf, x=pc_x, y=pc_y, ax=axs[0])
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[0].set_title('Scatter plot of the first two principal components')
        axs[1].plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
        axs[1].set_xlabel('Number of components')
        axs[1].set_ylabel('Cumulative explained variance')
        axs[1].set_title('Explained variance ratio')
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        
    correlations = np.corrcoef(hotel_rooms_scaled.transpose(), principalComponents.transpose())
    if circle:
        fig, ax = plt.subplots(figsize=(5,5))
        texts = []
        for i, label in enumerate(hotel_rooms.columns):
            ax.plot(correlations[i, pc_x], correlations[i, pc_y], 'r+')
            texts.append(ax.text(correlations[i, pc_x], correlations[i, pc_y], label))
        adjust_text(texts)
        circle = plt.Circle((0,0), 1, color='gray', fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f'PC{pc_x + 1}')
        ax.set_ylabel(f'PC{pc_y + 1}')
        ax.set_title('Correlation circle')
        plt.show()
    # i want to return the variables that are the most correlated with the two principal components pc_x and pc_y
    correlation_indices = np.argsort(correlations[:, pc_x] ** 2 + correlations[:, pc_y] ** 2)[::-1]
    valid_indices = correlation_indices[correlation_indices < len(hotel_rooms.columns)]
    return hotel_rooms.columns[valid_indices]

def plot_correlation_ranking(target,df=None, columns=None, except_columns=None, circle=False):
    if df is None:
        if columns is not None and len(columns)>0:
            if target not in columns:
                columns.append(target)
            df = generate_global_df(columns, except_columns=except_columns)
        else:
            raise ValueError('The columns list is empty')
    else:
        if columns is None:
            if target not in df.columns:
                raise ValueError('The target column is not in the dataframe')
            else:
                df = df
        else: df = df[columns]
    numeric_df = df.select_dtypes(include=[np.number])
    target_column = df[target]
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(numeric_df)
    pca = PCA()
    principalComponents = pca.fit_transform(scaled_df)

    correlations = []
    for i in range(principalComponents.shape[1]):
        pc = principalComponents[:, i]
        correlation = np.corrcoef(pc, target_column)[0, 1]
        correlations.append(correlation)

    most_correlated_pc = np.argmax(np.abs(correlations))
    correlations_abs = np.abs(correlations)
    correlations_abs[most_correlated_pc] = 0 
    second_most_correlated_pc = np.argmax(correlations_abs)
    
    # Plot 1: Variable Correlations
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    sorted_correlations_df = pd.DataFrame(correlations, columns=['Correlation'], index=numeric_df.columns)
    sorted_correlations_df.reset_index(inplace=True)
    sorted_correlations_df.columns = ['Variable', 'Correlation']
    sorted_correlations_df.sort_values('Correlation', inplace=True, ascending=False)
    colors = ['r' if x < 0 else 'b' for x in sorted_correlations_df['Correlation']]

    barplot = sns.barplot(x='Correlation', y='Variable', data=sorted_correlations_df, palette=colors, ax=ax2)
    ax2.set_title(f'Ranking of variable correlations with {target}')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Variables')
    
    # Annotate the bars with the three highest positive and negative correlation values
    top_positive = sorted_correlations_df.nlargest(3, 'Correlation')['Correlation'].values
    top_negative = sorted_correlations_df.nsmallest(3, 'Correlation')['Correlation'].values
    for p in barplot.patches:
        if p.get_width() in top_positive:
            barplot.annotate(format(p.get_width(), '.2f'), 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha = 'left', va = 'center', 
                            size=10,
                            xytext = (5, 0), 
                            textcoords = 'offset points')
        elif p.get_width() in top_negative:
            barplot.annotate(format(p.get_width(), '.2f'), 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha = 'right', va = 'center', 
                            size=10,
                            xytext = (-5, 0), 
                            textcoords = 'offset points')

    plt.show()
    if circle:
        # Plot 2: Circle of Correlations
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        pcs = pca.components_

        circle = plt.Circle((0,0), 1, color='k', fill=False)
        ax1.add_artist(circle)
        target_column_index = list(numeric_df.columns).index(target)

        x, y = pcs[most_correlated_pc, target_column_index], pcs[second_most_correlated_pc, target_column_index]
        ax1.arrow(0, 0, x, y, color='k')
        ax1.text(x, y, target, color='r', ha='center', va='center')
        ax1.set_xlim([-1,1])
        ax1.set_ylim([-1,1])
        ax1.set_xlabel(f"PC{most_correlated_pc+1}")
        ax1.set_ylabel(f"PC{second_most_correlated_pc+1}")
        ax1.set_title('Circle of Correlations')
        plt.show()

def plot_Evar_Hvar_correlations(df=None, only_hotel_vars=None, env_vars=None, display_all= False, show_abs = False, show_both=False):
    def compute_and_sort_corr(df, vars1, vars2):
        corr = df.corr().loc[vars1, vars2]
        corr_abs = df.corr().loc[vars1, vars2].abs()    
        #corr[vars2] = corr[vars2].abs().mean(axis=1)
        corr.sort_values(by=vars2, ascending=False, inplace=True)
        corr_abs.sort_values(by=vars2, ascending=False, inplace=True)
        #i need to add the variable names var1 for each row too for corr and corr_abs
        return corr, corr_abs

    def plot_heatmap(corr, ax, title, vmin, vmax, cbar=False, cbar_ax=None):
        sns.heatmap(corr, cmap='twilight_shifted', center=0,
                    square=True, linewidths=1, ax=ax,vmin=vmin, vmax=vmax, cbar=cbar, cbar_ax=cbar_ax)
        ax.set_title(title)
        

    if df is None:
        df = generate_global_df()

    
    df = df[only_hotel_vars + env_vars]
    corr1, corr1_abs = compute_and_sort_corr(df, env_vars, only_hotel_vars)
    corr2, corr2_abs = compute_and_sort_corr(df, only_hotel_vars, env_vars)
    vmin = min(corr1.min().min(), corr2.min().min())
    vmax = max(corr1.max().max(), corr2.max().max())
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 13.5))
    if not show_both:
        if not show_abs:
            plot_heatmap(corr1, axs[0], 'Correlation between Environment variables and Hotel variables', vmin, vmax, cbar=False)
            plot_heatmap(corr2, axs[1], 'Correlation between Hotel variables and Environment variables', vmin, vmax, cbar=True)
        else:
            plot_heatmap(corr1_abs, axs[0], 'Absolute Correlation between Environment variables and Hotel variables', vmin, vmax, cbar=False)
            plot_heatmap(corr2_abs, axs[1], 'Absolute Correlation between Hotel variables and Environment variables', vmin, vmax, cbar=True)
    else:
        plot_heatmap(corr1, axs[0], 'Correlation between Environment variables and Hotel variables', vmin, vmax, cbar=False)
        plot_heatmap(corr2, axs[1], 'Correlation between Hotel variables and Environment variables', vmin, vmax, cbar=False)
        #display the absolute values of the correlations in another figure
        fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(30, 13.5))
        plot_heatmap(corr1_abs, axs1[0], 'Absolute Correlation between Environment variables and Hotel variables', vmin, vmax, cbar=False)
        plot_heatmap(corr2_abs, axs1[1], 'Absolute Correlation between Hotel variables and Environment variables', vmin, vmax, cbar=False)
    plt.tight_layout()
    plt.show()
    if not display_all:
        corr1 = corr1.assign(env_vars = corr1.index)
        corr1_abs = corr1_abs.assign(env_vars = corr1_abs.index)
        return corr1, corr1_abs
    
    corr_df = corr1.unstack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    max_corr_df1 = corr_df.loc[corr_df.groupby('Variable 1')['Abs Correlation'].idxmax(), ['Variable 1', 'Variable 2', 'Correlation', 'Abs Correlation']].sort_values('Abs Correlation', ascending=False).reset_index(drop=True)
    max_corr_df2 = corr_df.loc[corr_df.groupby('Variable 2')['Abs Correlation'].idxmax(), ['Variable 2', 'Variable 1', 'Correlation', 'Abs Correlation']].sort_values('Abs Correlation', ascending=False).reset_index(drop=True)

    pos_corr_color = "#4586AC"
    neg_corr_color= "#CC5A49"

    pos_patch = mpatches.Patch(color=pos_corr_color, label='Positive Correlation')
    neg_patch = mpatches.Patch(color=neg_corr_color, label='Negative Correlation')

    fig1, ax1 = plt.subplots(figsize=(15, 10))
    barplot1 = sns.barplot(x='Abs Correlation', y='Variable 1', data=max_corr_df1, dodge=False, palette=[neg_corr_color if x < 0 else pos_corr_color for x in max_corr_df1['Correlation']], ax=ax1)
    ax1.set_title('For each Hotel variable, what is the Environmental variable that influences it the most?')
    for i, p in enumerate(barplot1.patches):
        width = p.get_width()
        color =  neg_corr_color if max_corr_df1['Correlation'][i] < 0 else pos_corr_color
        brightness = mcolors.rgb_to_hsv(mcolors.to_rgb(color))[2]
        text_color = 'white' if brightness < 0.5 else 'black'
        max_len = int(width*300)
        text = max_corr_df1['Variable 2'][i]
        ax1.text(0.5*p.get_width(), p.get_y()+0.55*p.get_height(),
                '{}'.format(text if len(text) < max_len else text[:max_len-3] + '...'),
                ha='center', va='center', color=text_color)
    plt.tight_layout()
    plt.legend(handles=[pos_patch, neg_patch])
    plt.show()

    # For each environmental variable, what is the hotel variable that influences it the most?
    fig2, ax2 = plt.subplots(figsize=(15, 30))
    barplot2 = sns.barplot(x='Abs Correlation', y='Variable 2', data=max_corr_df2, dodge=False, palette=[neg_corr_color if x < 0 else pos_corr_color for x in max_corr_df2['Correlation']], ax=ax2)
    ax2.set_title('For each Environmental variable, what is the Hotel variable that is influenced the most by it?')
    for i, p in enumerate(barplot2.patches):
        width = p.get_width()
        color = neg_corr_color if max_corr_df2['Correlation'][i] < 0 else pos_corr_color
        brightness = mcolors.rgb_to_hsv(mcolors.to_rgb(color))[2]
        text_color = 'white' if brightness < 0.5 else 'black'
        max_len = int(width * 300)
        text = max_corr_df2['Variable 1'][i]
        ax2.text(0.5*p.get_width(), p.get_y()+0.55*p.get_height(),
                '{}'.format(text if len(text) < max_len else text[:max_len-3] + '...'),
                ha='center', va='center', color=text_color)
    plt.tight_layout()
    plt.legend(handles=[pos_patch, neg_patch])
    plt.show()
    
    return corr1


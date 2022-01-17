import folium
from folium import plugins
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
from zipfile import ZipFile
import requests, io
from datetime import timedelta


from mobilipy import plot, preparation, waypointsdataframe, segmentation, mode_detection, legs, gtfs_helper, home_work, privacy


def plot_person_and_stops(person_latlon, stops_df, person_caption='Sample Zurich location'):
    m = folium.Map()

    folium.Marker(
        location=[person_latlon[0], person_latlon[1]],
        popup=person_caption,
        icon=folium.Icon(color='red', prefix='fa',icon='male'),
    ).add_to(m)

    for index, row in stops_df.iterrows():
        folium.Marker(
            location=[row['stop_lat'], row['stop_lon']],
            popup=row['stop_name'],
            icon=folium.Icon(color='blue', prefix='fa',icon='bus'),
    ).add_to(m)

    bounds = plot.get_map_bounds(stops_df.rename(columns={'stop_lon':'longitude', 'stop_lat':'latitude'}))
    m.fit_bounds(bounds)

    return m

@st.cache
def get_gps_data():
    data = pd.read_csv('geolife_sample.txt', compression='gzip')
    return data

@st.cache
def get_michal_data():
    data = pd.read_csv('https://raw.githubusercontent.com/plechoss/mobility/main/michal_waypoints.csv', delimiter=';')
    data['id'] = 1
    data['user_id'] = 1
    data = data[data['tracked_at']>='2021-11-24']
    data = data[(data['latitude']<52.360047) & (data['latitude']>52.032179)]
    data = data[(data['longitude']<21.411500) & (data['longitude']>20.729284)]
    
    return data

@st.cache
def get_gtfs_helper():
    zip_file_url = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrplandaten_gtfs/download/2021_google_transit.zip'
    r = requests.get(zip_file_url)
    z = ZipFile(io.BytesIO(r.content))
    z.extractall("gtfs")

    return gtfs_helper.GTFS_Helper(directory='./gtfs/')

@st.cache
def get_prepared_data(data):
    return preparation.prepare(data)
    
@st.cache
def get_legs(data_prepared):
    route_clusters_detected = segmentation.segment(data_prepared, use_multiprocessing=False)

    segmentation_stage.text('Detecting modes of transport...')
    route_clusters_detected = mode_detection.mode_detection(route_clusters_detected, use_multiprocessing=False)

    segmentation_stage.text('Getting legs...')
    legs_user = legs.get_user_legs(route_clusters_detected, '1', use_multiprocessing=False)
    legs_user = legs_user[legs_user.detected_mode.notnull()]

    segmentation_stage.text('')
    return legs_user


st.title('SDSC Mobility Package')
st.write('The library\'s purpose is to work with raw GPS data (called waypoints). It provides tools to clean the data, split it into trips, detect mode of transport as well as detect home and work locations. In addition to that, it offers a GTFS helper that makes working with GTFS files easier.')
st.markdown('''We will use sample data from examples in the [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility) library, located [here](https://github.com/scikit-mobility/scikit-mobility/blob/master/examples/geolife_sample.txt.gz).''')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

michal_data = get_michal_data()
#raw_data = get_gps_data()
raw_data = michal_data
data_length = raw_data.shape[0]
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

n_rows = st.number_input('Pick a number of waypoints to be loaded:', min_value=0, max_value=data_length, value=5000)

data = raw_data.head(n_rows)


#------------------------------------
st.subheader('Raw data')

st.markdown('''`data`''')
st.dataframe(data.head())

st.markdown('''First, we create a new `WaypointsDataFrame` object from the data.''')
st.code('''data = waypointsdataframe.WaypointsDataFrame(data, longitude='lng', latitude='lat', tracked_at='datetime', user_id='uid')''', language='python')

data = waypointsdataframe.WaypointsDataFrame(data, longitude='lng', latitude='lat', tracked_at='datetime', user_id='uid')

st.markdown('''`data`''')
st.dataframe(data.head())

m = plot.plot_gps(data, line=True)
folium_static(m)

#------------------------------------
st.header('Preparation')
st.markdown('''The `prepare(df)` method performs Gaussian smoothing on the waypoints and removes all points with `accuracy > 1000` if the column exists.''')
st.code('''data_prepared = preparation.prepare(data)''', language='python')
data_prepared = get_prepared_data(data)

st.markdown('''`data_prepared`''')
st.dataframe(data_prepared.head())

m = plot.plot_gps(data_prepared, line=True)
folium_static(m)


#------------------------------------
st.header('Trip splitting')

st.markdown('''Next, we perform segmentation and mode detection, and then we assemble the trip legs.''')
st.code('''
route_clusters_detected = segmentation.segment(data_prepared, use_multiprocessing=False)
route_clusters_detected = mode_detection.mode_detection(route_clusters_detected, use_multiprocessing=False)
legs_user = legs.get_user_legs(route_clusters_detected, '1', use_multiprocessing=False)
        ''', language='python')


segmentation_state = st.text('Splitting data...')

segmentation_stage = st.text('Segmenting...')

legs_user = get_legs(data_prepared)

st.markdown('''`legs_user`''')
st.dataframe(legs_user)

segmentation_state.text('Segmenting data...done!')

#------------------------------------
st.subheader('Plotting trip legs')

starting_timestamps = legs_user.started_at


#save selection
names = pd.DataFrame()
names['name'] = legs_user.started_at.dt.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + legs_user.detected_mode
names['started_at'] = legs_user.started_at
names = names.set_index('started_at')

get_name = lambda x: names.loc[x, 'name']

selected_timestamps = st.multiselect('Which legs do you want to see?', starting_timestamps, default=starting_timestamps[:5], format_func=get_name)
#filter legs df
selected_legs_user = legs_user[legs_user['started_at'].isin(selected_timestamps)]

#insert left column as name for the multiselect
selected_legs_user.insert(0, 'name', selected_legs_user.started_at.dt.strftime('%Y-%m-%d %H:%M:%S') + ' ' + selected_legs_user.detected_mode)

data_prepared.tracked_at = data_prepared.tracked_at.map(lambda x: x.replace(tzinfo=None))

m = plot.plot_legs(selected_legs_user, data_prepared)
folium_static(m)


#------------------------------------
st.header('Home and work detection')
st.markdown('''The home and work detection can be performed with the following line:''')
st.code('''home_location, work_location = home_work.detect_home_work(legs_user, data_prepared)''', language='python')
home_location, work_location = home_work.detect_home_work(legs_user, data_prepared)
st.markdown('''`legs_user`''')
st.dataframe(legs_user)

home = home_location
work = work_location

if(home is not None and work is not None):
    m = plot.plot_gps(data_prepared, line=True) 
    folium.Marker(
            location=home,
            popup='Home',
            icon=folium.Icon(color='blue', prefix='fa',icon='home'),
        ).add_to(m)
    folium.Marker(
            location=work,
            popup='Work',
            icon=folium.Icon(color='red', prefix='fa',icon='briefcase'),
        ).add_to(m)

    longitudes = [home[1], work[1]]
    latitudes = [home[0], work[0]]

    home_work_df = pd.DataFrame()
    home_work_df['longitude'] = longitudes
    home_work_df['latitude'] = latitudes

    bounds = plot.get_map_bounds(home_work_df)
    m.fit_bounds(bounds)

    folium_static(m)
else:
    st.write('No activities detected, can\'t show home and work locations!')

#------------------------------------
st.header('Privacy')
st.markdown('''The `privacy` module contains two functions that let the user privatize the dataset, namely `obfuscate` and `aggregate`''')
st.subheader('''`obfuscate`''')
st.write('This function obfuscates the area of a given radius around home and work locations by either removing all the points in these areas or changing them all to a single, noisy point in these areas.')
st.code('''obfuscated_df = privacy.obfuscate(data, [home, work], radius=1000, mode='remove')
m = plot.plot_gps(obfuscated_df, line=False)''', language='python')

map_loading_text = st.text('Obfuscating the data...')
radius = 1000
obfuscated_df, shifted_home, shifted_work = privacy.obfuscate(data, [home, work], radius=radius, mode='remove')
map_loading_text.text('Plotting the points...')
m = plot.plot_gps(obfuscated_df, line=True)
m.fit_bounds(bounds)
folium.Marker(
        location=home,
        popup='Home',
        icon=folium.Icon(color='blue', prefix='fa',icon='home'),
    ).add_to(m)
folium.Marker(
        location=work,
        popup='Work',
        icon=folium.Icon(color='red', prefix='fa',icon='briefcase'),
    ).add_to(m)
folium.Circle(
        location=shifted_home,
        radius=radius
    ).add_to(m)
folium.Circle(
        location=shifted_work,
        radius=radius
    ).add_to(m)

folium_static(m)
map_loading_text.text('')

st.subheader('''`aggregate`''')
st.write('This function groups the data into cells at a given time (with a timedelta span) and returns the number of users that appeared in that cell during the given timedelta. The returned dataframe contains no information about the user_ids.')
st.code('''from datetime import timedelta

delta = timedelta(hours=1)
cell_size = 0.2 #cell size in kilometers
aggregated_data = privacy.aggregate(data, cell_size, delta).reset_index()''',language='python')

delta = timedelta(hours=1)
cell_size = 0.2
aggregated_data = privacy.aggregate(data, cell_size, delta).reset_index()
st.dataframe(aggregated_data)

possible_times = aggregated_data.tracked_at.unique()
#save selection
selected_timestamp = st.selectbox('Pick a timestamp to see the heatmap:', possible_times, 0)
selected_agg_data = aggregated_data[aggregated_data['tracked_at']==selected_timestamp]
points = np.column_stack((selected_agg_data.cell_latitude.values, selected_agg_data.cell_longitude.values))

m = folium.Map(location=points[0])
hm = plugins.HeatMap(points)
m.fit_bounds(bounds)
m.add_child(hm)

folium_static(m)


#------------------------------------
st.header('GTFS Helper')
st.markdown('''The `GTFS_Helper` is a class that manages GTFS data and makes it easier to work with when it comes to mobility analysis. Helper functions for finding closests stops and stop transits were developed as a part of it.''')
gtfs_text = st.text('Initializing the GTFS Helper...')
st.code('''gtfs = gtfs_helper.GTFS_Helper()''', language='python')

gtfs = get_gtfs_helper()
gtfs_text.text('GTFS Helper initialized!')

st.text('''Input user's location in Zurich:''')
latitude = st.number_input('Latitude:', value=47.374665)
longitude = st.number_input('Longitude:', value=8.537343)

#------------------------------------
st.subheader('get_nearby_stops')
st.code('''stops = gtfs.get_nearby_stops(longitude=longitude, latitude=latitude)''', language='python')
stops = gtfs.get_nearby_stops(longitude=longitude, latitude=latitude)

st.markdown('''`stops`''')
st.dataframe(stops)

m = plot_person_and_stops((latitude, longitude), stops)
folium_static(m)


#------------------------------------
st.subheader('get_n_closest_stops')
st.code('''closest_stops = gtfs.get_n_closest_stops(longitude=longitude, latitude=latitude, n=3)''', language='python')
closest_stops = gtfs.get_n_closest_stops(longitude=longitude, latitude=latitude, n=3)

st.markdown('''`closest_stops`''')
st.dataframe(closest_stops)

m = plot_person_and_stops((latitude, longitude), closest_stops)
folium_static(m)

#------------------------------------
st.subheader('get_transfers')
st.code('''transfers = gtfs.transfers''')

transfers = gtfs.transfers
transfers = transfers.drop(columns=['neigh'])
st.markdown('''`transfers`''')
st.dataframe(transfers)

unique_stops = transfers[['stop_name', 'stop_id', 'stop_lat', 'stop_lon']].drop_duplicates()
unique_stops.insert(0, 'display_name', unique_stops['stop_name'] + ' ' + unique_stops['stop_id'])
unique_stops = unique_stops.sort_values(by=['display_name'])

#save selection
selected_stop = st.selectbox('Pick a stop to see its transfers:', unique_stops, 0)
stop_id = unique_stops[unique_stops['display_name']==selected_stop].iloc[0].stop_id
selected_transfers = transfers[transfers['stop_id'] == stop_id]

stop_latlon = (selected_transfers.iloc[0].stop_lat, selected_transfers.iloc[0].stop_lon)
neighbour_stops = selected_transfers[['neigh_name', 'neigh_lat', 'neigh_lon']].rename(columns={'neigh_name':'stop_name','neigh_lon':'stop_lon','neigh_lat':'stop_lat'})

m = plot_person_and_stops(stop_latlon, neighbour_stops, person_caption=selected_transfers.iloc[0].stop_name)
folium_static(m)

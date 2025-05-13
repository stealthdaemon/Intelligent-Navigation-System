# For high quality visualization
import matplotlib.pyplot as plt
import plotly_express as pe

# maps
import googlemaps

# Model maps as graphs with nodes and edges
import osmnx as ox
import networkx as nx

from datetime import datetime
import requests

pe.set_mapbox_access_token(
    'Your_Plotly_express_API_key')


def get_coordinates(graph, path):
    """Get coordinates of each node in a list"""
    coordinates = []
    graph_nodes = list(graph.nodes(data=True))
    for num in path:
        for n, details in graph_nodes:
            if num == n:
                coordinates.append((details['y'], details['x']))
                break
    return coordinates


def get_google_map_details(start_coordinates, dest_coordinates):
    """ Get google map direction details for origin and destination"""
    # Initialize Google Maps client
    api_key = 'Your_Google_Maps_API_key'
    gmaps = googlemaps.Client(key=api_key)

    # Define start and end locations
    ori = start_coordinates
    dest = dest_coordinates

    # Get directions between the origin and destination
    return gmaps.directions(ori, dest, mode='driving', departure_time=datetime.now(), alternatives=True)


def get_osm_map_details(start_coordinates, dest_coordinates):
    """ Get OSM direction details for origin and destination"""
    # Define start and end locations (lat, lon)
    origin_lat, origin_lon = start_coordinates
    dest_lat, dest_lon = dest_coordinates

    # OSRM API endpoint for driving directions
    osrm_api_url = 'http://router.project-osrm.org/route/v1/driving/'

    # API URL for the route request
    route_url = f"{osrm_api_url}{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson"

    # Get the route data from OSRM API
    return requests.get(route_url).json()


def get_osm_routes(direction_data):
    """Get coordinates for OSM map direction data"""
    coordinates = []
    for lon, lat in direction_data['routes'][0]['geometry']['coordinates']:
        coordinates.append((lat, lon))
    return coordinates


def get_google_coordinates(google_direction_data):
    """ Get coordinates for google maps direction data"""
    coordinates = []
    for leg in google_direction_data[1]['legs']:
        for step in leg['steps']:
            start_location = step['start_location']
            coordinates.append((start_location['lat'], start_location['lng']))

        # Ensure the end location is included
        end_location = leg['steps'][-1]['end_location']
        coordinates.append((end_location['lat'], end_location['lng']))

    return coordinates


def plot_maps(coordinates, alternate_route=None, title='Travel Path'):
    """Plot maps coordinates"""
    fig = pe.scatter_mapbox(
        lon=[v for u, v in coordinates],
        lat=[u for u, v in coordinates],
        zoom=12.5,
        width=800,
        height=600,
        animation_frame=list(range(0, len(coordinates))),
        title=title,
        mapbox_style="streets"
    )
    fig.data[0].marker = dict(size=12, color='black')

    # Adding the start point
    fig.add_trace(pe.scatter_mapbox(
        lon=[coordinates[0][1]],
        lat=[coordinates[0][0]]
    ).data[0])
    fig.data[1].marker = dict(size=15, color='green')

    # Adding the end point
    fig.add_trace(pe.scatter_mapbox(
        lon=[coordinates[-1][1]],
        lat=[coordinates[-1][0]]
    ).data[0])
    fig.data[2].marker = dict(size=15, color='red')

    # Adding the line for the route
    fig.add_trace(pe.line_mapbox(
        lon=[v for u, v in coordinates],
        lat=[u for u, v in coordinates]
    ).data[0])

    if alternate_route:
        fig.data[3].line = dict(color="red", width=2)
        # Adding the alternate route
        fig.add_trace(pe.line_mapbox(
            lon=[v for u, v in alternate_route],
            lat=[u for u, v in alternate_route]
        ).data[0])
        fig.data[-1].line = dict(color="blue", width=2)
    return fig


def compute_weight(graph, path, weight):
    """Computes the total specified weight for a given set of nodes."""
    total = 0.0

    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        # Add the 'length' of the edge between current_node and next_node
        total += graph[current_node][next_node][0].get(weight, 0)

    return total


def show_plots(G, r=None, t=None, c=None, o=None, d=None, default=None):
    # Show route, traffic and road closure nodes

    # routes, traffic nodes, and road closure nodes
    routes = r
    traffic_nodes = t
    road_closure_nodes = c
    origin = o
    dest = d

    # Plot the graph
    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor='#ffffff',
        node_color='grey',
        figsize=(12, 12)
    )

    if routes is not None:
        # Highlight the route in blue
        ox.plot_graph_route(
            G,
            routes,
            route_linewidth=4,
            route_color='blue',
            route_alpha=0.7,
            node_color='none',
            ax=ax,
            show=False,
            close=False
        )

    if default:
        ox.plot_graph_route(
            G,
            default,
            route_linewidth=2,
            route_color='red',
            route_alpha=0.3,
            node_color='none',
            ax=ax,
            show=False,
            close=False
        )

    if traffic_nodes:
        # Highlight traffic nodes in yellow
        nx.draw_networkx_nodes(
            G,
            pos=ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.map(lambda x: (x.x, x.y)),
            nodelist=traffic_nodes,
            node_color='yellow',
            node_size=50,
            ax=ax
        )

    if road_closure_nodes:
        # Highlight road closure nodes in purple
        nx.draw_networkx_nodes(
            G,
            pos=ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.map(lambda x: (x.x, x.y)),
            nodelist=road_closure_nodes,
            node_color='purple',
            node_size=50,
            ax=ax
        )

    if origin:
        # Highlight Origin node in green
        nx.draw_networkx_nodes(
            G,
            pos=ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.map(lambda x: (x.x, x.y)),
            nodelist=origin,
            node_color='green',
            node_size=50,
            ax=ax
        )

    if dest:
        # Highlight destination node in red
        nx.draw_networkx_nodes(
            G,
            pos=ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.map(lambda x: (x.x, x.y)),
            nodelist=dest,
            node_color='red',
            node_size=50,
            ax=ax
        )

    # Show the plot
    plt.show()

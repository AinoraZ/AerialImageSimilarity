from __future__ import annotations
import base64
import numpy as np
from io import BytesIO
import math
from map_provider import MapProvider
from vector import Vector2D
from plotly import graph_objects 
from PIL import Image

class MapPlotter:
    def __init__(self, map_provider: MapProvider, graph_size=1500):
        self.map_size = map_provider.image_size
        self.padding = math.floor(max(self.map_size.x, self.map_size.y) * 0.005)

        self.graph_dimensions = self._calculate_graph_dimensions(graph_size)
        self.map_base64 = self._base64_background(map_provider)

    def _base64_background(self, map_provider: MapProvider) -> str:
        im_file = BytesIO()

        map_image = Image.fromarray(map_provider.image_np)
        map_image.resize((self.graph_dimensions.x, self.graph_dimensions.y), Image.ANTIALIAS).save(im_file, format="JPEG")
        encoded_string = base64.b64encode(im_file.getvalue()).decode()
        
        return "data:image/png;base64," + encoded_string

    def _calculate_graph_dimensions(self, largest_dimension):
        if self.map_size.x >= self.map_size.y:
            graph_width = largest_dimension
            graph_height = int(self.map_size.y * largest_dimension / self.map_size.x)
        else:
            graph_width = int(self.map_size.x * largest_dimension / self.map_size.y)
            graph_height = largest_dimension
            
        return Vector2D(graph_width, graph_height)

    def plot_graph(self, particles, title, weights=None, drone_x=None, drone_y=None, predicted_x=None, predicted_y=None):
        if weights is None:
            weights = [4 for _ in range(len(particles))]
        else:
            weights = [max(4, weight) for weight in weights]

        #name='Dalelės',
        graph_data = [graph_objects.Scattergl(x=particles[:,0], y=particles[:,1], marker=dict(size=weights), mode='markers')]

        if drone_x != None and drone_y != None:
            #name='Tikroji vieta', 
            graph_data.append(graph_objects.Scattergl(x=[drone_x], y=[drone_y],  marker=dict(size=[14]), mode='markers'))

        if predicted_x != None and predicted_y != None:
            # name='Spėjama vieta', 
            graph_data.append(graph_objects.Scattergl(x=[predicted_x], y=[predicted_y], marker=dict(size=[8]), mode='markers'))

        fig = graph_objects.Figure(data=graph_data)

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font=dict(family="Times New Roman", size=12), showlegend=False)
        fig.update_layout(width = self.graph_dimensions.x, height = self.graph_dimensions.y) # , title = title
        fig.update_xaxes(range=[0 - self.padding, self.map_size.x + self.padding], autorange = False, visible=False, showticklabels=False) #title = 'x'
        fig.update_yaxes(range=[0 - self.padding, self.map_size.y + self.padding], autorange = False, visible=False, showticklabels=False) #title = 'y'
        fig.add_layout_image(
            source=self.map_base64,
            xref="x",
            yref="y",
            x=0,
            y=self.map_size.y,
            sizex=self.map_size.x,
            sizey=self.map_size.y,
            sizing="stretch",
            opacity=0.75,
            layer="below")        

        return fig
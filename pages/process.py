import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import app

# Load data
pickleFile = open("assets/clean_mushroom_data.pkl", 'rb')
mushrooms = pickle.load(pickleFile)
pickleFile.close()

# Load feature embeddings
pickleFile = open("assets/feature_embeddings.pkl", 'rb')
X_embedded = pickle.load(pickleFile)
pickleFile.close()

# Format data for visualization
y = mushrooms['class'].replace({'p':0, 'e':1})
data = pd.DataFrame(X_embedded)
data['class'] = y
data.columns = ['comp_1', 'comp_2', 'class']
poison = data[data['class'] == 0]
edible = data[data['class'] == 1]

# Create figure
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=edible["comp_1"],
    y=edible["comp_2"],
    name = "Edible",
    mode='markers',
    marker=dict(
        color='rgba(124, 250, 149, .5)',
        line_width=1
    )
))
fig.add_trace(go.Scattergl(
    x=poison["comp_1"],
    y=poison["comp_2"],
    name = "Poison",
    mode='markers',
    marker=dict(
        color='rgba(160, 65, 201, .5)',
        line_width=1
    )
))

fig.update_layout(title="T-SNE Visualization",
                  yaxis_zeroline=False, xaxis_zeroline=False)

column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## Process


            """
        ),
        dcc.Graph(figure=fig),

    ],
)

layout = dbc.Row([column1])

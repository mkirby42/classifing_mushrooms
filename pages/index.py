import pickle
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go


from app import app

"""
https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

Layout in Bootstrap is controlled using the grid system. The Bootstrap grid has
twelve columns.

There are three main layout components in dash-bootstrap-components: Container,
Row, and Col.

The layout of your app should be built as a series of rows of columns.

We set md=4 indicating that on a 'medium' sized or larger screen each column
should take up a third of the width. Since we don't specify behaviour on
smaller size screens Bootstrap will allow the rows to wrap so as not to squash
the content.
"""

column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## Edible or Poisonous?

            The Lepitoa genus of mushrooms can be difficult to label as either edible or Poisonous.

            We've used scikit-learn to develop a model which can classify Lepitoa mushrooms as
            edible or poisonous with astounding precision based on some easily
            observable physical characteristics.

            The figure on the right utilizes t-distributed stochastic neighbor
            embeddings (T-SNE) to create a 3-dimensional representation of
            over 90 different physical characteristics from a randomly sampled
            subset of our data.
            """
        ),
        dcc.Link(dbc.Button('See More', color='primary'), href='/models')
    ],
    md=4,
)

# Load data
pickleFile = open("assets/clean_mushroom_data.pkl", 'rb')
mushrooms = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open("assets/3d_feature_embeddings.pkl", 'rb')
embeddings = pickle.load(pickleFile)
pickleFile.close()

y = mushrooms['class'].replace({'p':0, 'e':1})

data = pd.DataFrame(embeddings)
data['class'] = y
data.columns = ['comp_1', 'comp_2', 'comp_3', 'class']
data = data.sample(frac = .1)

poison = data[data['class'] == 0]
edible = data[data['class'] == 1]

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=poison['comp_1'], y=poison['comp_2'], z=poison['comp_3'],
        name = 'Poison',
        mode='markers',
        marker=dict(
            size=4,
            color='#ff005e',
            opacity=0.9,
            line=dict(
                    color='rgba(0, 0, 0, 0.5)',
                    width=1,
            )
        ),
    )
)

fig.add_trace(go.Scatter3d(
    x=edible['comp_1'], y=edible['comp_2'], z=edible['comp_3'],
    name = 'Edible',
    mode='markers',
    marker=dict(
        size=4,
        color='#00ffe8',
        opacity=0.9,
        line=dict(
                color='rgba(0, 0, 0, 0.5)',
                width=1,
        )
    )
))

camera = dict(
    up=dict(x=0, y=1, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=1, z=.05)
)


fig.update_layout(showlegend = True)
fig.update_layout(scene_camera = camera)


column2 = dbc.Col(
    [
        dcc.Graph(figure = fig, config = {'displayModeBar': True}),
    ]
)

layout = dbc.Row([column1, column2])

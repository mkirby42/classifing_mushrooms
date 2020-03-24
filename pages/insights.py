import pickle
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from app import app

# Load extra layouts
cyto.load_extra_layouts()

def generate_cyto_elements(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    elements = []
    for node in range(n_nodes):
        if feature[node] == -2:
          lab = 'Decision'
        else:
          lab = f"X[:, {feature[node]}] <= {threshold[node]}"

        elements.append({"data": {"id": node, "label": lab},
                         "position": {"x": np.random.randint(low = 0, high = 200),
                                      "y": np.random.randint(low = 0, high = 200)},
                         "classes": 'node_color'})

        if children_left[node] != -1:
            elements.append({"data": {"source": node, "target":children_left[node]}})
        if children_right[node] != -1:
            elements.append({"data": {"source": node, "target":children_right[node]}})

    return elements


# Load model
pickleFile = open("mushroom_rfmodel.pkl", 'rb')
best_model = pickle.load(pickleFile)
pickleFile.close()


column1 = dbc.Col(
    [
        
    ],
    md=4,
)


# Create cytoscape elements
decision_tree_elements = generate_cyto_elements(np.random.choice(best_model.estimators_))

column2 = dbc.Col(
    [
html.Div([
    cyto.Cytoscape(
        id='cytoscape-layout-4',
        elements=decision_tree_elements,
        style={'width': '100%', 'height': '800px'},
        layout={
            'name': 'dagre'
        },
        stylesheet=[
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                }
            },

            # Class selectors
            {
                'selector': '.node_color',
                'style': {
                    'background-color': 'red',
                    'line-color': 'black'
                }
            }
        ]
    )
])
    ]
)


layout = dbc.Row([column1, column2])

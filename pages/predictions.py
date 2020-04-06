import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_cytoscape as cyto
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.model_selection import train_test_split

from app import app
cyto.load_extra_layouts()


def load_pickle(filepath):
    pickleFile = open(filepath, 'rb')
    obj = pickle.load(pickleFile)
    pickleFile.close()
    return obj


def generate_cyto_elements(estimator, data, sample_index = 0):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    node_indicator = estimator.decision_path(data)
    decision_nodes = node_indicator.indices[node_indicator.indptr[sample_index]:
                                    node_indicator.indptr[sample_index + 1]]

    elements = []
    for node in range(n_nodes):
        if (data.values[sample_index, feature[node]] <= threshold[node]):
            threshold_sign = "False"
        else:
            threshold_sign = "True"

        # Format decision string
        if (feature[node] == -2) & (node in decision_nodes):
            sample = data.iloc[sample_index].values.reshape(1, -1)
            lab = str(estimator.predict(sample))
        elif feature[node] == -2:
            lab = ' '
        else:
            lab = "%s %s (%s)" %(list(data.columns)[feature[node]],
                                  threshold_sign,
                                  data.values[sample_index, feature[node]]
                                  )

        # Edit classes for nodes in decision path
        if node in decision_nodes:
            class_labels = "decision_node"
        else:
            class_labels = 'node_color'

        # Create node data
        elements.append({"data": {"id": node, "label": lab},
                         "position": {"x": np.random.randint(low = 0, high = 200),
                                      "y": np.random.randint(low = 0, high = 200)},
                         "classes": class_labels})

        # Create children pointer data
        if children_left[node] != -1:
            elements.append({"data": {"source": node, "target":children_left[node]}})
        if children_right[node] != -1:
            elements.append({"data": {"source": node, "target":children_right[node]}})

    return elements

# Load models
models = {}
models['decision_stump'] = load_pickle("assets/decision_stump.pkl")
models['vanilla_decision_tree'] = load_pickle("assets/vanilla_decision_tree.pkl")
models['opto_decision_tree'] = load_pickle("assets/opto_decision_tree.pkl")
models['vanilla_forest'] = load_pickle("assets/vanilla_forest.pkl")
models['opto_forest'] = load_pickle("assets/opto_forest.pkl")

X_test = load_pickle("assets/X_test.pkl")
y_test = load_pickle("assets/y_test.pkl")


# Format data for display
show_df = X_test.copy()
show_df['Prediction'] = models['opto_decision_tree'].predict(X_test)
show_df['Prediction'] = show_df['Prediction'].replace({0:'Poison', 1:'Edible'})
show_df['Class'] = y_test.replace({0:'Poison', 1:'Edible'})
show_df['Prediction Correct ?'] = show_df.apply(lambda row: row.Class == row.Prediction, axis = 1)
show_df = show_df.reset_index()
show_df = show_df[['index', 'Class', 'Prediction', 'Prediction Correct ?']]


column1 = dbc.Col(
    [
    dcc.Markdown(
        """

        # Decision Tree Visualization

        With a decision stump, a decision tree with a node depth of one, we can identify the single feature which is most useful for our classification task: no odor.
        With decision trees of increasing depth our decision paths become more complex and our model is able to classify more accurately.

        """
    ),
    html.Div(
        id = 'div_1',
        style={'marginBottom': 25, 'marginTop': 25}
    ),
    html.Button('Generate New Random Test Set Sample', id = 'button'),
    ], width = 'auto',
)


column2 = dbc.Col(
    [
    html.Div(
        id = 'div_1',
        style={'marginBottom': 25, 'marginTop': 25}
    ),
    dcc.Markdown(
        """#### Select Model"""
    ),
    dcc.Dropdown(
        id = 'model_selection_dropdown',
        options=[
            {'label': 'Decision Stump', 'value': 'decision_stump'},
            {'label': 'Decision Tree (Depth: 2)', 'value': 'vanilla_decision_tree'},
            {'label': 'Decision Tree (Depth: 3)', 'value': 'vanilla_forest'},
            {'label': 'Decision Tree (Depth: 4)', 'value': 'opto_forest'},
            {'label': 'Optimized Decision Tree', 'value': 'opto_decision_tree'},
        ],
        value= 'decision_stump'
    ),
    html.Div(
        id = 'div_2',
        style={'marginBottom': 25, 'marginTop': 25}
    ),
    cyto.Cytoscape(
            id='cytoscape-layout-4',
            elements=generate_cyto_elements(
                models['decision_stump'],
                X_test,
                0,
                ),
            style={'width': '100%', 'height': '400px'},
            layout={
                'name': 'dagre',
                'spacingFactor': '1'
            },
            stylesheet=[
                # Group selectors
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'text-halign': 'left',
                        'font-size': '10'
                    }
                },

                # Class selectors
                {
                    'selector': '.node_color',
                    'style': {
                        'background-color': 'red',
                        'line-color': 'black'
                    }
                },
                {
                    'selector': '.decision_node',
                    'style': {
                        'background-color': 'blue',
                        'line-color': 'black'
                    }
                }
            ],
        ),
    ]
)


# Update Cytoscape elements callback
@app.callback(
    Output('cytoscape-layout-4','elements'),
    [Input('button', 'n_clicks'),
     Input('model_selection_dropdown', 'value')],
    [State('cytoscape-layout-4', 'elements')])
def f(selected_rows, value, elements):
    if selected_rows:
        elements = generate_cyto_elements(
            models[value],
            X_test,
            numpy.random.randint(0, len(X_test)),
        )
    else:
        elements = generate_cyto_elements(
            models[value],
            X_test,
            0,
        )
    return elements



layout = dbc.Row([column1, column2])

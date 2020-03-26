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
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        # Format decision string
        if feature[node] == -2:
          lab = 'Decision'
        else:
          lab = "%s %s %s (%s)" %(list(data.columns)[feature[node]],
                                  threshold_sign,
                                  threshold[node],
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
show_df['prediction'] = models['opto_decision_tree'].predict(X_test)
show_df['prediction'] = show_df['prediction'].replace({0:'Poison', 1:'Edible'})
show_df['class'] = y_test.replace({0:'Poison', 1:'Edible'})
show_df = show_df.reset_index()
show_df = show_df[['index', 'class', 'prediction']]


column1 = dbc.Col(
    [
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
            {'label': 'Optomized Decision Tree', 'value': 'opto_decision_tree'},
        ],
        value= 'decision_stump'
    ),
    html.Div(
        id = 'div_1',
        style={'marginBottom': 25, 'marginTop': 25}
    ),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in show_df.columns],
        data=show_df.to_dict('records'),
        style_table={
            'maxHeight': '300px',
            'overflowY': 'scroll'
        },
        fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={'width': '60px'},
        row_selectable="single",
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 9,
    ),
    ],
)

@app.callback(
    [Output('table', 'columns'),
    Output('table', 'data')],
    [Input('model_selection_dropdown', 'value')])
def update_output(value):
    data = show_df.drop(columns = ['prediction'])
    data['prediction'] = models[value].predict(X_test)
    cols = [{"name": i, "id": i} for i in data.columns]
    return cols, data.to_dict('records')

column2 = dbc.Col([html.Div([
    cyto.Cytoscape(
        id='cytoscape-layout-4',
        elements=generate_cyto_elements(
            models['decision_stump'],
            X_test,
            0,
            ),
        style={'width': '100%', 'height': '300px'},
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
            },
            {
                'selector': '.decision_node',
                'style': {
                    'background-color': 'blue',
                    'line-color': 'black'
                }
            }
        ]
    )
])])


@app.callback(
    Output('cytoscape-layout-4','elements'),
    [Input('table', 'rows'),
     Input('table', 'selected_rows'),
     Input('model_selection_dropdown', 'value')],
    [State('cytoscape-layout-4', 'elements')])
def f(rows, selected_rows, value, elements):
    if len(selected_rows) > 0:
        elements = generate_cyto_elements(
            models[value],
            X_test,
            selected_rows[0],
        )
    else:
        elements = generate_cyto_elements(
            models[value],
            X_test,
            0,
        )
    return elements


layout = dbc.Row([column1, column2])

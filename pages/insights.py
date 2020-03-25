import pickle
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.model_selection import train_test_split

from app import app

# Load extra layouts
cyto.load_extra_layouts()

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


# Load model
pickleFile = open("assets/vanilla_decision_tree.pkl", 'rb')
best_model = pickle.load(pickleFile)
pickleFile.close()

# Load data
pickleFile = open("assets/clean_mushroom_data.pkl", 'rb')
mushrooms = pickle.load(pickleFile)
pickleFile.close()

# Format features, targets
X = mushrooms.drop(columns='class')
X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X)
y = mushrooms['class'].replace({'p':0, 'e':1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                    test_size=.2, stratify=y)


column1 = dbc.Col(
    [
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in mushrooms.columns],
        data=mushrooms.to_dict('records'),
        style_table={
            'maxHeight': '500px',
            'overflowY': 'scroll'
        },
        fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={'width': '75px'},
        row_selectable="single",
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    )
    ],
    md=4,
)

# Create cytoscape elements
decision_tree_elements = generate_cyto_elements(
    best_model,
    X_test,
    np.random.choice(range(len(X_test))),
    )


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
    ]),
    ]
)

layout = dbc.Row([column1, column2])

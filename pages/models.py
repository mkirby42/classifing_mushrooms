import pickle
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
import dash_table

from app import app


def load_pickle(filepath):
    pickleFile = open(filepath, 'rb')
    obj = pickle.load(pickleFile)
    pickleFile.close()
    return obj


decision_stump = load_pickle("assets/decision_stump.pkl")
vanilla_decision_tree = load_pickle("assets/vanilla_decision_tree.pkl")
opto_decision_tree = load_pickle("assets/opto_decision_tree.pkl")
vanilla_forest = load_pickle("assets/vanilla_forest.pkl")
opto_forest = load_pickle("assets/opto_forest.pkl")
X_test = load_pickle("assets/X_test.pkl")
y_test = load_pickle("assets/y_test.pkl")


predictions = {}
predictions['decision_stump'] = decision_stump.predict(X_test)
predictions['vanilla_decision_tree'] = vanilla_decision_tree.predict(X_test)
predictions['opto_decision_tree'] = opto_decision_tree.predict(X_test)
predictions['vanilla_forest'] = vanilla_forest.predict(X_test)
predictions['opto_forest'] = opto_forest.predict(X_test)


def get_confusion_matrix(model_predictions):
    con_matrix = pd.DataFrame(confusion_matrix(y_test, model_predictions),
        columns=['Predicted Poison',
                 'Predicted Edible'],
        index=['Actual Poison',
               'Actual Edible']
        ).reset_index()
    con_matrix.rename(columns={'index':''}, inplace=True)
    return con_matrix


def get_class_report(model_predictions):
    report = pd.DataFrame(classification_report(
            y_test,
            model_predictions,
            target_names=['0-Poisonous', '1-Edible'],
            output_dict = True)
        ).round(2).reset_index()
    report.rename(columns={'index':''}, inplace=True)
    return report


column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            # Model Evolution

            We can use decision trees to classify mushrooms based on physical characteristics such as gill size, stalk length, and spore print color.

            These models utilize precision as their performance metric because we wish to minimize false positives for the edible class, that is poisonous mushrooms classified as edible.

            """
        ),
        html.Div(
            id = 'div_1',
            style={'marginBottom': 25, 'marginTop': 25}
        ),
        html.Div(
            id = 'div_1',
            style={'marginBottom': 25, 'marginTop': 25}
        ),
        dcc.Link(dbc.Button('Visualize Tree Models', color='primary'), href='/predictions')
    ],
    style={'marginBottom': 50, 'marginTop': 25}
)

# Confusion Matrix Callback
@app.callback(
    [Output('conf_matrix', 'columns'),
    Output('conf_matrix', 'data')],
    [Input('model_selection_dropdown', 'value')])
def update_output(value):
    data = get_confusion_matrix(predictions[value])
    cols = [{"name": i, "id": i} for i in data.columns]
    return cols, data.to_dict('records')


# Classification Report Callback
@app.callback(
    [Output('class_report', 'columns'),
    Output('class_report', 'data')],
    [Input('model_selection_dropdown', 'value')])
def update_output(value):
    data = get_class_report(predictions[value])
    cols = [{"name": i, "id": i} for i in data.columns]
    return cols, data.to_dict('records')


column2 = dbc.Col(
    [
        html.Div(
            id = 'div_3',
            style={'marginBottom': 10, 'marginTop': 10}
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
        dcc.Markdown(
            """

            #### Confusion Matrix

            """
        ),
        dash_table.DataTable(
            id='conf_matrix',
            columns=[{"name": i, "id": i} for i in get_confusion_matrix(
                decision_stump.predict(X_test)).columns],
            data=get_confusion_matrix(
                decision_stump.predict(X_test)).to_dict('records'),
        ),
        html.Div(
            id = 'div_3',
            style={'marginBottom': 25, 'marginTop': 25}
        ),
        dcc.Markdown(
            """

            #### Classification Report

            """
        ),
        dash_table.DataTable(
            id='class_report',
            columns=[{"name": i, "id": i} for i in get_class_report(
                decision_stump.predict(X_test)).columns],
            data=get_class_report(
                decision_stump.predict(X_test)).to_dict('records'),
        ),
    ],
    style={'marginBottom': 50, 'marginTop': 25}
)

layout = dbc.Row([column1, column2])

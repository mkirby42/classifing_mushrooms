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

from app import app

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

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
    return con_matrix

def get_class_report(model_predictions):
    report = pd.DataFrame(classification_report(
            y_test,
            model_predictions,
            target_names=['0-Poisonous', '1-Edible'],
            output_dict = True)
        ).round(2).reset_index()
    return report


column1 = dbc.Col(
    [
        dcc.Markdown(
        """

        ## Predictions


        """
        ),
        html.Label('Select Model'),
        dcc.Dropdown(
            id = 'model_selection_dropdown',
            options=[
                {'label': 'Decision Stump', 'value': 'decision_stump'},
                {'label': 'Vanilla Decision Tree', 'value': 'vanilla_decision_tree'},
                {'label': 'Optomized Decision Tree', 'value': 'opto_decision_tree'},
                {'label': 'Vanilla Random Forest', 'value': 'vanilla_forest'},
                {'label': 'Optomized Random Forest', 'value': 'opto_forest'}
            ],
            value= 'decision_stump'
        ),
    ],
    style={'marginBottom': 50, 'marginTop': 25}
)

@app.callback(
    dash.dependencies.Output('conf_matrix', 'children'),
    [dash.dependencies.Input('model_selection_dropdown', 'value')])
def update_output(value):
    return generate_table(get_confusion_matrix(predictions[value]))

@app.callback(
    dash.dependencies.Output('class_report', 'children'),
    [dash.dependencies.Input('model_selection_dropdown', 'value')])
def update_output(value):
    return generate_table(get_class_report(predictions[value]))

column2 = dbc.Col(
    [
        dcc.Markdown("""## Confusion Matrix"""),
        html.Div(
            id = 'conf_matrix',
            children=[generate_table(get_confusion_matrix(decision_stump.predict(X_test)))],
            style={'marginBottom': 50, 'marginTop': 25}),

        dcc.Markdown("""## Classification Report"""),
        html.Div(
            id = 'class_report',
            children=[generate_table(get_class_report(decision_stump.predict(X_test)))],
            style={'marginBottom': 50, 'marginTop': 25})
    ]
)

layout = dbc.Row([column1, column2])

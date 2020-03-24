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

# Load model
pickleFile = open("mushroom_rfmodel.pkl", 'rb')
best_model = pickle.load(pickleFile)
pickleFile.close()

# Load data
pickleFile = open("clean_mushroom_data.pkl", 'rb')
mushrooms = pickle.load(pickleFile)
pickleFile.close()

# Format features, targets
X = mushrooms.drop(columns='class')
X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X)
y = mushrooms['class'].replace({'p':0, 'e':1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                    test_size=.2, stratify=y)

model_predictions = best_model.predict(X_test)

con_matrix = pd.DataFrame(confusion_matrix(y_test, model_predictions),
    columns=['Predicted Poison', 'Predicted Edible'],
    index=['Actual Poison', 'Actual Edible'])

con_matrix = con_matrix.reset_index()


column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## Predictions


            """
        ),
        html.Div(children=[
            generate_table(con_matrix)
        ])
    ],
    md=4,
)

column2 = dbc.Col(
    [

    ]
)

layout = dbc.Row([column1, column2])

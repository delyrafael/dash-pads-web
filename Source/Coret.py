import dash
from dash import Dash, dash_table, html, dcc, dbc
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('car_predict.csv', delimiter=";")
df = df.dropna()

app = Dash(__name__)
button = html.Button('Tombol aja')

app.layout = html.Div(
    # style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},
    children=[
        html.H1('Data Prediksi Mobil'),  # Menambahkan judul sebagai elemen H1
        dash_table.DataTable(
            id='datatable',
            data=df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_current=0,
            page_size=10,  # Jumlah baris per halaman
            page_action='native',  # Menggunakan paginasi bawaan
            style_table={'width': '500px', 'height': '300px', 'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        ),
        
        dbc.Button('Test Button'),
    
        html.Div(id='prediction-output'),
        html.Div([
            html.Button('Prediksi K-Means', id='kmeans-prediction-button', n_clicks=0),
            html.Button('Prediksi Decision Tree', id='dt-prediction-button', n_clicks=0),
        ], style={'marginTop': '20px'}),
        dcc.Input(id='kmeans-input', type='number', placeholder='Masukkan input untuk K-Means'),
        html.Button('Submit K-Means', id='kmeans-submit-button', n_clicks=0),
        html.Div(id='kmeans-output'),
        dcc.Input(id='dt-input', type='number', placeholder='Masukkan input untuk Decision Tree'),
        html.Button('Submit Decision Tree', id='dt-submit-button', n_clicks=0),
        html.Div(id='dt-output')
        
    ],
    
    
)


@app.callback(
    Output('kmeans-output', 'children'),
    Input('kmeans-submit-button', 'n_clicks'),
    State('kmeans-input', 'value')
)
def make_kmeans_prediction(n_clicks, input_value):
    if n_clicks > 0 and input_value is not None:
        # Melatih model K-Means
        kmeans_model = KMeans(n_clusters=3)
        kmeans_model.fit(df.drop('target_column', axis=1))

        prediction = kmeans_model.predict([[input_value]])
        return f'Prediksi K-Means: {prediction[0]}'
    else:
        return ''


@app.callback(
    Output('dt-output', 'children'),
    Input('dt-submit-button', 'n_clicks'),
    State('dt-input', 'value')
)
def make_dt_prediction(n_clicks, input_value):
    if n_clicks > 0 and input_value is not None:
        # Melatih model Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(df.drop('target_column', axis=1), df['target_column'])

        prediction = dt_model.predict([[input_value]])
        return f'Prediksi Decision Tree: {prediction[0]}'
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)

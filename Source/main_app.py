from dash import Dash, dash_table
from dash.dependencies import Input, Output

import dash_core_components as dcc
import pandas as pd
import dash_table
from dash import html


df = pd.read_csv('car_predict.csv', delimiter=";")
df = pd.DataFrame(df)
df = df.dropna()


app = Dash(__name__)

# PAGE_SIZE = 5

    
app.layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},  
    children=[
        html.H1('Car Prediction Data'),  # Add the title as an H1 element
        dash_table.DataTable(
            id='datatable',
            data=df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            # style_table={'width': '500px', 'height': '300px'}  # Set the desired width and height
            page_current=0,
            page_size=10,  # Number of rows per page
            page_action='native',  # Use the built-in pagination
            style_table={'width': '500px', 'height': '300px', 'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ]
)



@app.callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"))
def update_table(page_current,page_size):
    return df.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)

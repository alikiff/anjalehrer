from dash import Dash, html, dcc, callback, Output, Input


# Initialize the Dash app
app = Dash(__name__)
server = app.server

# Create a simple layout
app.layout = html.Div([
    html.H1('My First Dash App'),
    dcc.Dropdown(
        id='dropdown',
        options=['Option 1', 'Option 2', 'Option 3'],
        value='Option 1'
    ),
    html.Div(id='output-text')
])

# Add a simple callback
@callback(
    Output('output-text', 'children'),
    Input('dropdown', 'value')
)
def update_output(value):
    return f'You selected: {value}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
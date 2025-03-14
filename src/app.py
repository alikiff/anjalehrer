from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
from pathlib import Path
import base64
import os
import tempfile
import helper_ocr_llm

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Create layout with Bootstrap components
app.layout = dbc.Container([
    html.H1('Vocabulary Extractor', className='text-center mb-4'),
    
    dbc.Card([
        dbc.CardBody([
            dcc.Upload(
                id='upload-images',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select up to 5 Images')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),
            html.Div(id='output-image-upload', className='mt-3'),
            html.Div(id='vocabulary-output', className='mt-4')
        ])
    ])
], fluid=True)

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as fp:
        fp.write(base64.b64decode(data))
        return fp.name

@app.callback(
    Output('vocabulary-output', 'children'),
    Input('upload-images', 'contents'),
    State('upload-images', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is None:
        return html.Div("Upload some images to start...")
    
    all_vocabularies = []
    
    try:
        # Process only up to 5 images
        for content, name in zip(list_of_contents[:5], list_of_names[:5]):
            # Save temporary file
            temp_path = save_file(name, content)
            
            # Process image using helper functions
            base64_image = helper_ocr_llm.encode_image(temp_path)
            api_key = helper_ocr_llm.get_mistral_api_key_from_dot_env(
                Path(".").resolve() / "src" / ".env"
            )
            client = helper_ocr_llm.get_mistral_client(api_key)
            ocr_response = helper_ocr_llm.ocr_get_text(client, base64_image)
            messages = helper_ocr_llm.llm_helper_create_messages(ocr_response)
            vocabularies_raw = helper_ocr_llm.llm_extract_vocabulary(client, messages)
            vocabularies = helper_ocr_llm.format_get_vocabulary_list(vocabularies_raw)
            
            all_vocabularies.extend(vocabularies)
            
            # Clean up temporary file
            os.unlink(temp_path)
        
        # Convert vocabularies to table format
        table_data = []
        for vocab_dict in all_vocabularies:
            for eng, ger in vocab_dict.items():
                table_data.append({'English': eng, 'German': ger})
        
        return dbc.Card([
            dbc.CardHeader("Extracted Vocabularies"),
            dbc.CardBody(
                dash_table.DataTable(
                    data=table_data,
                    columns=[
                        {'name': 'English', 'id': 'English'},
                        {'name': 'German', 'id': 'German'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            )
        ])
        
    except Exception as e:
        return html.Div(f'Error processing images: {str(e)}')

# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)
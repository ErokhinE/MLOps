import gradio as gr
import mlflow
from model import load_features
from transform_data import transform_data
import json
import requests
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from data import transform
def make_config():
    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name='main')
    return cfg

cfg = make_config()

# You need to define a parameter for each column in your raw dataset
def predict(year = None,
            make = None,
            model = None,
            trim = None,
            body = None,
            transmission = None,
            vin = '5xyktca69fg566472',
            state = None,
            condition = None,
            odometer = None,
            color = None,
            interior = None,
            seller = None,
            mmr = None,
            sellingprice = None,
            saledate = 'Tue Dec 16 2014 12:30:00 GMT-0800 (PST)'):
    
    # This will be a dict of column values for input data sample
    features = {
        "year": year, 
        "make": make, 
        "model": model, 
        "trim": trim, 
        "body": body, 
        "transmission": transmission, 
        "vin": vin,
        "state": state,
        "condition": condition,
        "odometer": odometer,
        "color": color,
        "interior": interior,
        "seller": seller,
        "mmr": mmr,
        "sellingprice": sellingprice,
        "saledate": saledate
    }
    
    print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X, _ = transform_data(raw_df)
    # Convert it into JSON
    example = X.iloc[0,:]

    example = json.dumps( 
        { "features": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{cfg.flask_port}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    to_return = response.json()['prediction'][0]
    return to_return

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will be populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Number(label = "year"), 
        gr.Text(label="make"),
        gr.Text(label="model"),
        gr.Text(label="trim"),
        gr.Text(label="body"),
        gr.Text(label="transmission"),   
        gr.Text(label = "vin"), 
        gr.Text(label="state"),   
        gr.Number(label="condition"),   
        gr.Number(label="odometer"),
        gr.Text(label="color"),
        gr.Text(label="interior"),
        gr.Text(label="seller"),
        gr.Number(label="mmr"),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="prediction result"),
    
    # This will provide the user with examples to test the API
    # examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# Launch the web UI locally on port 5155
demo.launch(server_port = cfg.web_ui_port)

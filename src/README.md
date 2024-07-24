# Directory for python scripts

src
├── app.py                    #gradio webinterface
├── data_prepare.py           #zenml pipeline to preprocess the data
├── data.py                   #most basic and essential functions
├── evaluate.py               #evaluates models
├── expectations_initial_dataset.py  #great expectations 
├── extract_charts.py         #extratc metric charts for each model and its version
├── main.py                   #train and save models in mlflow 
├── model.py                  #functions to train and save models in mlflow
├── predict.py                #uses model in a docker container to get predictions
├── transform_data.py         #transforms data to not use pipeline
├── validate_champion.py      #validates champion model for CI/CD pipeline
└── validate.py               #validates all versions of challengers and selects the best
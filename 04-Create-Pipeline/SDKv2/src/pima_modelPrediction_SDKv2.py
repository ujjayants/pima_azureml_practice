import os
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
#sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
from azureml.core import Run, Model


if __name__ == '__main__':

    # run = Run.get_context()

    # arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, help="output data from previous step is used as input")
    parser.add_argument("--output",     type=str, help="output directory")
    args = parser.parse_args()

    # input file path
    input_folder_path = os.path.join(args.processed_data)
    
    # getting data
    for f in os.listdir(input_folder_path):
        if 'processed_data' in f:
            input_file_path = os.path.join(input_folder_path, f)
            processed_df = pd.read_csv(input_file_path)
            # df, processed_df = joblib.load(input_file_path)

        if 'original_data' in f:
            input_file_path = os.path.join(input_folder_path, f)
            df = pd.read_csv(input_file_path)
            # df, processed_df = joblib.load(input_file_path)


    # Head & shape
    print(df.head(20))
    print('DataFrame shape:', df.shape)
    

    # Loading Model: Method 1
    #model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'credit_defaults_model.pkl')
    #model =  joblib.load(model_path)

    # Loading Model: Method 2

    model_name = 'pima_pipeline_model_SDKv2_03'

    model_path = Model.get_model_path(model_name)

    model = mlflow.sklearn.load_model(model_path)
    print('Loaded Trained Model:', model)

    # Prediction
    pred = model.predict(processed_df)

    # Append to original df
    df['pred']  = pred


    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)
    
    # Stop Logging
    mlflow.end_run()

    
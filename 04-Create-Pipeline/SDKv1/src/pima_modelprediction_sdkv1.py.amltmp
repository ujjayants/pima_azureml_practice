# import libraries
#Azure
from azureml.core import Run, Model
#python
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

if __name__ == '__main__':

    run = Run.get_context()

    # arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, help="output data from previous step is used as input")
    parser.add_argument("--output",     type=str, help="output directory")
    args = parser.parse_args()

    # input file path
    input_folder_path = os.path.join(args.processed_data)
    
    # getting data
    for f in os.listdir(input_folder_path):

        if 'df_original' in f:
            input_file_path = os.path.join(input_folder_path, f)
            df = pd.read_csv(input_file_path)
            
        if 'processed_df' in f:
            input_file_path = os.path.join(input_folder_path, f)
            processed_df = pd.read_csv(input_file_path)
            #df, processed_df = joblib.load(input_file_path)


    # Head & shape
    print(df.head(20))
    print('DataFrame shape:', df.shape)
    

    # Loading Model: Method 1
    #model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'credit_defaults_model.pkl')
    #model =  joblib.load(model_path)

    # Loading Model: Method 2
    ws = run.experiment.workspace

    model_name = 'pima_pipeline_model_SDKv1_04'
    model_obj  = Model(ws, name=model_name) # by default takes the latest version
    artifacts_path = model_obj.download(exist_ok = True)

    #### Get paths for each artifact
    model_path = os.path.join(artifacts_path ,f'{model_name}.pkl')    
    print(model_path)

    model =  joblib.load(model_path)
    print('Loaded Trained Model:', model)

    # Prediction
    pred = model.predict(processed_df)

    # Append to original df
    df['pred']  = pred

    # Save it as CSV or store it in a sql table as per your requirement
    # for simplicity storing as csv in pipeline output folder
    # saving data
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created output folder: {args.output}")

    save_path  = os.path.join(args.output , f'predictions.csv')
    df.to_csv(save_path, header=True, index=False)

    run.complete()

    

    

    
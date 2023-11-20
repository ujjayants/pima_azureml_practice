# import libraries
#Azure
from azureml.core import Run
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
    parser.add_argument("--actual_prediction_data", type=str, help="output data from previous step is used as input")
    parser.add_argument("--output",     type=str, help="output directory")
    args = parser.parse_args()

    # input file path
    input_folder_path = os.path.join(args.actual_prediction_data)
    
    # getting data
    for f in os.listdir(input_folder_path):
        if 'df_val' in f:
            input_file_path = os.path.join(input_folder_path, f)
            df = pd.read_csv(input_file_path)
    
        elif 'pima_pipeline_model' in f:
            model_file_path= os.path.join(input_folder_path, f)
            #model, scaler, unique_train_values, column_mapping = joblib.load(model_file_path)
            model = joblib.load(model_file_path)
            print('Model:',model)

        elif 'preproc_artifacts' in f:
            preproc_artifacts_path= os.path.join(input_folder_path, f)
            preproc_artifacts = joblib.load(preproc_artifacts_path)
            print('preproc_artifacts:', preproc_artifacts)
            
            scaler = preproc_artifacts['scaler']
            unique_values_train = preproc_artifacts['unique_values_train']
            column_mapping =  preproc_artifacts['column_mapping']

        # elif 'scaler' in f:
        #     scaler_file_path= os.path.join(input_folder_path, f)
        #     scaler = joblib.load(scaler_file_path)
        #     print('Scaler:', scaler)

        # elif 'unique_values_train' in f:
        #     cat_schema_file_path= os.path.join(input_folder_path, f)
        #     unique_values_train = joblib.load(cat_schema_file_path)
        #     print('Categorical schema:', unique_values_train)

        # elif 'column_mapping' in f:
        #     column_mapping_file_path= os.path.join(input_folder_path, f)
        #     column_mapping = joblib.load(column_mapping_file_path)
        #     print('Column_mapping:', column_mapping)



    # Head & shape
    print(df.head(20))
    print('DataFrame shape:', df.shape)
    
    # Metrics
    print('Confusion Matrix')
    print(confusion_matrix(df['Actual'], df['Prediction']))
    
    # to log to azure artifacts
    accuracy = accuracy_score(df['Actual'], df['Prediction'])
    run.log('Accuracy', accuracy)

    # Save the trained model in the azure artifacts outputs folder
    dir_path = 'outputs'
    os.makedirs(dir_path, exist_ok=True)

    model_name = 'pima_pipeline_model_SDKv1_04'
    model_path  = f'{dir_path}/pima_pipeline_model_SDKv1_04.pkl'
    scaler_path = f'{dir_path}/scaler.pkl'
    categorical_path = f'{dir_path}/unique_values_train.pkl'
    column_mapping_path = f'{dir_path}/column_mapping.pkl'

    # dump    
    joblib.dump(value=model, filename=model_path)
    joblib.dump(value= scaler, filename=scaler_path)
    joblib.dump(value=unique_values_train, filename= categorical_path)
    joblib.dump(value= column_mapping, filename= column_mapping_path)

    # upload
    print('uploading model to experiment run')
    run.upload_file(model_path, model_path)
    run.upload_file(scaler_path, scaler_path)
    run.upload_file(categorical_path, categorical_path)
    run.upload_file(column_mapping_path, column_mapping_path)


    ###############


    print('registering model')
    run.register_model(model_name=model_name,
                       model_path= dir_path,  
                       description='Pima diabetes Detection pipeline Model using SDKv1'
                      )
    
    run.complete()

    

    

    
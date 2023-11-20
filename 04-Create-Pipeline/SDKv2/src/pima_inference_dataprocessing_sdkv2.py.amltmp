# Importing Libraries
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow
import joblib

from azureml.core import Run,Model


## Use this if I save the values used during imputation while training
'''
def impute_mv(df, impute_dict):

    print( 'Actual input columns- ',df_train.columns)
    # Calculate the median value for BMI
    median_bmi = imput_dict['BMI']
    # Substitute it in the BMI column of the
    # dataset where values are 0
    
    df['BMI'] = df['BMI'].replace(
        to_replace=0, value=median_bmi)

    median_bloodp = impute_dict['BloodPressure']
    # Substitute it in the BloodP column of the
    # dataset where values are 0
    df['BloodPressure'] = df['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)
    

    # Calculate the median value for PlGlcConc
    median_plglcconc = impute_dict['Glucose']
    # Substitute it in the PlGlcConc column of the
    # dataset where values are 0
    df['Glucose'] = df['Glucose'].replace(
        to_replace=0, value=median_plglcconc)


    # Calculate the median value for SkinThick
    median_skinthick = impute_dict['SkinThickness']
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df['SkinThickness'] = df['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)


    # Calculate the median value for SkinThick
    median_skinthick = impute_dict['Insulin']
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df['Insulin'] = df['Insulin'].replace(
        to_replace=0, value=median_skinthick)
        
    return df
'''


def set_bmi(row):
    if row["BMI"] < 18.5:
        return "Under"
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "Healthy"
    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "Over"
    elif row["BMI"] >= 30:
        return "Obese"


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


def inference_preprocessing(df):

    # Data pre-processing artifacts
    # _, scaler, unique_train_values, column_mapping = artifacts
        
    ## add column re-mapping part
    df = df.rename(columns= { key:value for key, value in column_mapping.items() if value != 'Outcome'  })
    
    # BMI
    df = df.assign(BM_DESC= df.apply(set_bmi, axis=1))

    # Change to type categorical
    df['BM_DESC'] = df['BM_DESC'].astype('category')
 
    # Insulin
    df = df.assign(INSULIN_DESC= df.apply(set_insulin, axis=1))
 
    # Change to type categorical
    df['INSULIN_DESC'] = df['INSULIN_DESC'].astype('category')
 

    for col in df.columns:
        if col in unique_values_train:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            df[col] = df[col].astype('category').cat.set_categories(unique_values_train[col])


    # One-hot encoding
    cols_drop=['Insulin','BMI']
    df = df.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    df = pd.get_dummies(df)

    # Transform the data using the computed minimum and maximum values
    df_num_scaled = scaler.transform(df[numeric_columns])

    df_scaled = pd.concat([  pd.DataFrame(df_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  df[ df.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )


    del df, df_num_scaled

    return df_scaled


def load_artifacts():

    try:

        global scaler, unique_values_train, column_mapping

        model_path = Model.get_model_path('pima_pipeline_model_SDKv2_03')
        print('MODEL PATH', model_path)

        scaler = joblib.load(model_path + '/scaler.pkl')
        print(scaler)
        
        unique_values_train = joblib.load(model_path + '/unique_values_train.pkl')
        print(unique_values_train)
        
        column_mapping = joblib.load(model_path + '/column_mapping.pkl')
        print(column_mapping)

        logging.info("Init complete")
        
    except Exception as e:

        print('Exception occured in init():', e)


if __name__ == "__main__":

    #Initializing run object
    # run = Run.get_context()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="pipeline step output directory")

    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)

    # loading data
    df = pd.read_csv(args.data)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features",df.shape[1] - 1)

    print('EWWEWEWEWEWEWEW')
    ## Load artifacts
    load_artifacts()

    ##### Prep-processing

    processed_df = inference_preprocessing(df) #, artifacts)

    print(df.columns)
    print(processed_df.columns)


    # save_path = os.path.join(args.output, "inference_processed_data.pkl")

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    df.to_csv(os.path.join(args.output, "original_data.csv"), index=False)
    processed_df.to_csv(os.path.join( args.output, "processed_data.csv"), index=False)

    # joblib.dump(value= [df,processed_df], filename= save_path)

    # Stop Logging
    mlflow.end_run()
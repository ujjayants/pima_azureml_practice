# import libraries
#Azure
from azureml.core import Run, Model
#python
import os
import pandas as pd
import numpy as np
import argparse
import warnings
import sklearn
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


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

        # Load artifacts
        try:
            
            global scaler, unique_values_train, column_mapping

            # Loading Model: Method 2
            ws = run.experiment.workspace
            model_name = 'pima_pipeline_model_SDKv1_04'
            model_obj  = Model(ws, name=model_name) # by default takes the latest version
            artifacts_path = model_obj.download(exist_ok = True)

            #### Get paths for each artifact

            scaler_path = os.path.join(artifacts_path, 'scaler.pkl')
            print(scaler_path)

            unique_train_values_path = os.path.join(artifacts_path, 'unique_values_train.pkl')
            print(unique_train_values_path)

            column_mapping_path = os.path.join(artifacts_path, 'column_mapping.pkl')
            print(column_mapping_path)

            #### Extract artifacts            
            scaler = joblib.load(scaler_path)
            unique_values_train = joblib.load(unique_train_values_path)
            column_mapping = joblib.load(column_mapping_path)
 
            print('ARTIFACTS LOADED')     

        except Exception as e:
            print('Exception occured:', e)
        finally:
            pass



if __name__=='__main__':

    #Initializing run object
    run = Run.get_context()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="pipeline step output directory")

    args = parser.parse_args()

    # Loading input data   
    df = run.input_datasets['raw_data'].to_pandas_dataframe()

    # Load artifacts- model and preprocessing items
    load_artifacts()

    ##### Prep-processing

    processed_df = inference_preprocessing(df) #, artifacts)

    print(df.columns)
    print(processed_df.columns)

    # saving data
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created inference pipeline step output folder: {args.output}")

    save_path  = os.path.join(args.output , f'df_original.csv')
    save_path2  = os.path.join(args.output , f'processed_df.csv')

    # save both the original and processed df
    # joblib.dump(value= [df,processed_df ], filename= save_path)

    df.to_csv(save_path, header=True, index=False)
    
    processed_df.to_csv(save_path2, header=True, index=False)


    run.complete()
   
    
     
    

    

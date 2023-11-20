import json
import joblib
import numpy as np
import os
from azureml.core import Model
import pandas as pd

# Called when the service is loaded
def init():
    
    try:
        global model, scaler, unique_train_values, column_mapping
        # Method 1 : pickle name
        print(os.getenv("AZUREML_MODEL_DIR"))
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'outputs/pima_model_SDKv1_03.pkl')
        print(model_path)

        scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'outputs/scaler.pkl')
        print(scaler_path)

        unique_train_values_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'outputs/unique_values_train.pkl')
        print(unique_train_values_path)

        column_mapping_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'outputs/column_mapping.pkl')
        print(column_mapping_path)

        # Method 2 : Model Name
        # model_path = Model.get_model_path('credit_defaults_model')

        model= joblib.load(model_path)        
        print('model loaded')
        
        scaler= joblib.load(scaler_path)
        unique_train_values = joblib.load(unique_train_values_path)
        column_mapping = joblib.load(column_mapping_path)

        print(model)
        print(scaler)
        print(unique_train_values)
        print(column_mapping)

    except Exception as e:
        print('Exception occured:', e)
    finally:
        pass

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

def data_pred_prep(json_input):

    df_json  =  pd.DataFrame(json_input, 
    columns = [value for key,value in column_mapping.items() if value != 'Outcome'])

    # BMI
    df_json = df_json.assign(BM_DESC= df_json.apply(set_bmi, axis=1))

    # Change to type categorical
    df_json['BM_DESC'] = df_json['BM_DESC'].astype('category')

    # Insulin
    df_json = df_json.assign(INSULIN_DESC= df_json.apply(set_insulin, axis=1))

    # Change to type categorical
    df_json['INSULIN_DESC'] = df_json['INSULIN_DESC'].astype('category')

    for col in df_json.columns:
        if col in unique_train_values:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            df_json[col] = df_json[col].astype('category').cat.set_categories(unique_train_values[col])


    # One-hot encoding
    cols_drop=['Insulin','BMI']
    df_json = df_json.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = df_json.select_dtypes(include=['number']).columns.tolist()

    df_json = pd.get_dummies(df_json)

    # Transform the data using the computed minimum and maximum values
    df_json_num_scaled = scaler.transform(df_json[numeric_columns])

    df_json_scaled = pd.concat([  pd.DataFrame(df_json_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  df_json[ df_json.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )


    del df_json, df_json_num_scaled

    return df_json_scaled


# Called when a request is received
def run(raw_data):

    print('run function called')
    data = json.loads(raw_data)["data"]

    data = np.array( data_pred_prep(data)    )
    result = model.predict(data)
    print("Request processed")
    return result.tolist()

    
    
import os
import glob
import mlflow
import joblib
import pandas as pd
from azureml.core import Model
import logging

def init():

    try:

        global model, scaler, unique_train_values, column_mapping

        model_path = Model.get_model_path('pima_model_SDKv2_04')
        model = joblib.load(model_path + '/model.pkl')
        print(model)

        scaler = joblib.load(model_path + '/scaler.pkl')
        print(scaler)
        
        unique_train_values = joblib.load(model_path + '/unique_values_train.pkl')
        print(unique_train_values)
        
        column_mapping = joblib.load(model_path + '/column_mapping.pkl')
        print(column_mapping)

        logging.info("Init complete")
        
    except Exception as e:

        print('Exception occured in init():', e)


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


def data_pred_prep(df_input):

    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')

    # Rename columns using the rename() function
    df_input = df_input.rename(columns= { key:value for key, value in column_mapping.items() if value != 'Outcome'  })

    # BMI
    df_input = df_input.assign(BM_DESC= df_input.apply(set_bmi, axis=1))

    # Change to type categorical
    df_input['BM_DESC'] = df_input['BM_DESC'].astype('category')

    # Insulin
    df_input = df_input.assign(INSULIN_DESC= df_input.apply(set_insulin, axis=1))

    # Change to type categorical
    df_input['INSULIN_DESC'] = df_input['INSULIN_DESC'].astype('category')

    for col in df_input.columns:
        if col in unique_train_values:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            df_input[col] = df_input[col].astype('category').cat.set_categories(unique_train_values[col])


    # One-hot encoding
    cols_drop=['Insulin','BMI']
    df_input = df_input.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = df_input.select_dtypes(include=['number']).columns.tolist()

    df_input = pd.get_dummies(df_input)

    # Transform the data using the computed minimum and maximum values
    df_input_num_scaled = scaler.transform(df_input[numeric_columns])

    df_input_scaled = pd.concat([  pd.DataFrame(df_input_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  df_input[ df_input.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )


    del df_input, df_input_num_scaled

    return df_input_scaled




def run(mini_batch):

    try:
        print('mini_batch:',mini_batch)
        for file_path in mini_batch:
            # loading data
            df = pd.read_csv(file_path)
            print('DataFrame Shape:', df.shape)

            # prediction
            pred= model.predict( data_pred_prep(df) )
            df['Pred'] = pred
            
            return df

    except Exception as e:
        print('Exceptione Occured in run():', e)

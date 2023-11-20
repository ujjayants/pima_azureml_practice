# Importing Libraries
import os
import pandas as pd
import joblib
from azureml_user.parallel_run import EntryScript
from azureml.core import Model, Run


    
def init():
    
        try:
            
            global model, scaler, unique_train_values, column_mapping

            # Loading Model: Method 2
            run= Run.get_context()
            ws = run.experiment.workspace
            model_name = 'pima_model_SDKv1_03'
            model_obj  = Model(ws, name=model_name) # by default takes the latest version
            artifacts_path = model_obj.download(exist_ok = True)

            #### Get paths for each artifact
            model_path = os.path.join(artifacts_path ,f'{model_name}.pkl')    
            scaler_path = os.path.join(artifacts_path, 'scaler.pkl')
            unique_train_values_path = os.path.join(artifacts_path, 'unique_values_train.pkl')
            column_mapping_path = os.path.join(artifacts_path, 'column_mapping.pkl')

            #### Extract artifacts
            model =  joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            unique_train_values = joblib.load(unique_train_values_path)
            column_mapping = joblib.load(column_mapping_path)
 
            print(scaler)
            print(unique_train_values)
            print(column_mapping)
            print('ARTIFACTS LOADED')     

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


def data_pred_prep(df_input):

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



def run(df):

    try:
        results = []

        # prediction
        pred = model.predict(  data_pred_prep(df)  )
        df['Prediction'] = pred

        # appending to list
        results.append(df)
        
        return results

    except Exception as e:
        print("Exception Occured in run():", e)



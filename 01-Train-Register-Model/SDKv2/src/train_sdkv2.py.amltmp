# Importing Libraries
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
import os
import pandas as pd
import mlflow
import logging
import joblib

def impute_mv(df_train, df_val):

    # Calculate the median value for BMI
    median_bmi = df_train['BMI'].median()
    # Substitute it in the BMI column of the
    # dataset where values are 0
    df_train['BMI'] = df_train['BMI'].replace(
        to_replace=0, value=median_bmi)
    
    df_val['BMI'] = df_val['BMI'].replace(
        to_replace=0, value=median_bmi)

    median_bloodp = df_train['BloodPressure'].median()
    # Substitute it in the BloodP column of the
    # dataset where values are 0
    df_train['BloodPressure'] = df_train['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)
    
    df_val['BloodPressure'] = df_val['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)

    # Calculate the median value for PlGlcConc
    median_plglcconc = df_train['Glucose'].median()
    # Substitute it in the PlGlcConc column of the
    # dataset where values are 0
    df_train['Glucose'] = df_train['Glucose'].replace(
        to_replace=0, value=median_plglcconc)

    df_val['Glucose'] = df_val['Glucose'].replace(
        to_replace=0, value=median_plglcconc)


    # Calculate the median value for SkinThick
    median_skinthick = df_train['SkinThickness'].median()
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df_train['SkinThickness'] = df_train['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)

    df_val['SkinThickness'] = df_val['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)


    # Calculate the median value for SkinThick
    median_skinthick = df_train['Insulin'].median()
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    df_train['Insulin'] = df_train['Insulin'].replace(
        to_replace=0, value=median_skinthick)
    
    df_val['Insulin'] = df_val['Insulin'].replace(
        to_replace=0, value=median_skinthick)
    
    return df_train, df_val


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



if __name__ == "__main__":


    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--train_test_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()


    # Loading processed data
    data_file = os.path.join(args.input_data)   
    df = pd.read_csv(data_file)
    
    column_mapping = { df.columns[0]:'Pregnancies',df.columns[1] :'Glucose',
    df.columns[2]:'BloodPressure', df.columns[3] : 'SkinThickness',
    df.columns[4]: 'Insulin' , df.columns[5]:  'BMI',
    df.columns[6] : 'DiabetesPedigreeFunction' , df.columns[7] : 'Age' ,
    df.columns[8] :'Outcome'}

    # Rename columns using the rename() function
    df.rename(columns=column_mapping, inplace=True)

    # Split data

    X_train, X_val, y_train, y_val = train_test_split(df.loc[:, df.columns != 'Outcome'], df['Outcome'], test_size = args.train_test_ratio, stratify= df['Outcome'], random_state = 1234)

    X_train, X_val = impute_mv(X_train, X_val)

    #df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))

    # BMI
    X_train = X_train.assign(BM_DESC= X_train.apply(set_bmi, axis=1))
    X_val = X_val.assign(BM_DESC= X_val.apply(set_bmi, axis=1))

    # Change to type categorical
    X_train['BM_DESC'] = X_train['BM_DESC'].astype('category')
    X_val['BM_DESC'] = X_val['BM_DESC'].astype('category')


    # Insulin
    X_train = X_train.assign(INSULIN_DESC= X_train.apply(set_insulin, axis=1))
    X_val = X_val.assign(INSULIN_DESC= X_val.apply(set_insulin, axis=1))

    # Change to type categorical
    X_train['INSULIN_DESC'] = X_train['INSULIN_DESC'].astype('category')
    X_val['INSULIN_DESC'] = X_val['INSULIN_DESC'].astype('category')

    # Identify unique categorical values in the training set
    unique_values_train =  {col: X_train[col].unique() for col in X_train.select_dtypes(include=['category']).columns}

    for col in X_val.columns:
        if col in unique_values_train:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            X_val[col] = X_val[col].astype('category').cat.set_categories(unique_values_train[col])


    # One-hot encoding
    cols_drop=['Insulin','BMI']
    X_train=X_train.drop(cols_drop,axis=1)
    X_val=X_val.drop(cols_drop,axis=1)

    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()

    X_train= pd.get_dummies(X_train)
    X_val= pd.get_dummies(X_val)

    ############################################

    # Initialize the StandardScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data (computes the minimum and maximum values)
    scaler.fit(X_train[numeric_columns])

    # Train data- Transform the data using the computed minimum and maximum values
    X_train_num_scaled = scaler.transform(X_train[numeric_columns])

    X_train_scaled = pd.concat([  pd.DataFrame(X_train_num_scaled, columns= numeric_columns).reset_index(drop= True)
    ,  X_train[ X_train.columns.difference(numeric_columns)  ].reset_index(drop= True)  ] , axis =1  )

    # Val data- Transform the data using the computed minimum and maximum values
    X_val_num_scaled = scaler.transform(X_val[numeric_columns])

    X_val_scaled = pd.concat([pd.DataFrame(X_val_num_scaled, columns= numeric_columns).reset_index(drop= True)  ,
            X_val[ X_val.columns.difference(numeric_columns)  ].reset_index(drop= True) ], axis = 1)

    del X_train, X_val, X_train_num_scaled, X_val_num_scaled


    # Training Model
    model = RandomForestClassifier(class_weight='balanced',
                                    bootstrap=True,
                                    max_depth=100,
                                    max_features=2,
                                    min_samples_leaf=5,
                                    min_samples_split=10,
                                    n_estimators=1000,
                                    random_state = 42
                                    )
    
    # with Sampling
    # smote = SMOTE()
    # X_smote, y_smote = smote.fit_resampe(X_train, y_train)
    # model.fit(X_smote, y_smote)

    # without Sampling
    model.fit(X_train_scaled, y_train)
    y_pred= model.predict(X_val_scaled)

    print(classification_report(y_val, y_pred))

    # Save model and other artifacts
    dir_path = 'outputs'
    os.makedirs(dir_path, exist_ok=True)

    model_name = args.registered_model_name #'pima_model_SDKv2_03'
    # model_path  = f'{dir_path}/pima_model_SDKv2_0.pkl'
    scaler_path = f'{dir_path}/scaler.pkl'
    categorical_path = f'{dir_path}/unique_values_train.pkl'
    column_mapping_path = f'{dir_path}/column_mapping.pkl'

    # dump    
    #joblib.dump(value=model, filename=model_path)
    joblib.dump(value= scaler, filename=scaler_path)
    joblib.dump(value=unique_values_train, filename= categorical_path)
    joblib.dump(value= column_mapping, filename= column_mapping_path)

    # upload
    # newly added
    #mlflow.log_artifacts(dir_path)

    # Registering Model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model( sk_model= model,
                              registered_model_name= args.registered_model_name,
                              artifact_path= dir_path
                            )
    # prev- artifact_path = dir_path

    # args.registered_model_name
    # /mnt/azureml/cr/j/0045c96d883d46868f266f16097e5d44/cap/data-capability/wd/model

    # Saving the model to a file
    # mlflow.sklearn.save_model( sk_model= model,
    #                            path=os.path.join(args.model, "model"),
    #                          )    
    

    # Stop Logging
    mlflow.end_run()




    
           
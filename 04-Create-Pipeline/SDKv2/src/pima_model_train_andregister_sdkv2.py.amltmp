# Importing Libraries
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

import os
import pandas as pd
import mlflow
import logging


if __name__ == "__main__":

    #Initializing run object
    # run = Run.get_context()
    
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to processed data")
    # parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # getting list of dir
    files_list = os.listdir(args.input_data)
    print(f'Previous Step Output: {files_list}')

    # input file path
    input_folder_path = os.path.join(args.input_data)
    print(f"Training Data folder path:{input_folder_path}")
    
    # Loading processed data
    for f in os.listdir(input_folder_path):
        if 'processed_data' in f:
            data_file = os.path.join(input_folder_path, f)
            df = pd.read_csv(data_file)
            
            # # load pickle HERE
            # df = joblib.load(data_file)

        elif 'column_mapping' in f:
            data_file = os.path.join(input_folder_path, f)

            # load pickle HERE
            column_mapping = joblib.load(data_file)
            print(column_mapping)


    # Split Data
    X_train = df[df.split_type == 'Training']
    X_val = df[df.split_type == 'Val']

    y_train = X_train.Outcome.reset_index(drop= True)
    y_val = X_val.Outcome.reset_index(drop= True)

    ### NEWLY ADDED
    X_train = X_train.drop(columns = ['split_type']).reset_index(drop= True)
    X_val = X_val.drop(columns = ['split_type']).reset_index(drop= True)

    # Change to type categorical
    X_train['BM_DESC'] = X_train['BM_DESC'].astype('category')
    X_val['BM_DESC'] = X_val['BM_DESC'].astype('category')

    # Change to type categorical
    X_train['INSULIN_DESC'] = X_train['INSULIN_DESC'].astype('category')
    X_val['INSULIN_DESC'] = X_val['INSULIN_DESC'].astype('category')

    # Identify unique categorical values in the training set
    unique_values_train =  {col: X_train[col].unique() for col in X_train.select_dtypes(include=['category']).columns}

    for col in X_val.columns:
        if col in unique_values_train:
            # Ensure that the validation set's categorical column has the same unique values as the training set
            X_val[col] = X_val[col].astype('category').cat.set_categories(unique_values_train[col])

    ###
    
    X_train = X_train.drop(columns = ['Insulin','BMI','Outcome']).reset_index(drop= True)

    X_val = X_val.drop(columns = ['Insulin','BMI','Outcome']).reset_index(drop= True)

    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()

    # One-hot encoding
    X_train= pd.get_dummies(X_train)
    X_val= pd.get_dummies(X_val)


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

    model.fit(X_train_scaled, y_train)
    y_pred= model.predict(X_val_scaled)
    print(classification_report(y_val, y_pred))


    # Save model and other artifacts
    dir_path = 'outputs'
    os.makedirs(dir_path, exist_ok=True)

    model_name = args.registered_model_name #'pima_model_SDKv2_03'
    # model_path  = f'{dir_path}/pima_model_SDKv2_03.pkl'
    scaler_path = f'{dir_path}/scaler.pkl'
    categorical_path = f'{dir_path}/unique_values_train.pkl'
    column_mapping_path = f'{dir_path}/column_mapping.pkl'

    # dump    
    #joblib.dump(value=model, filename=model_path)
    joblib.dump(value= scaler, filename=scaler_path)
    joblib.dump(value=unique_values_train, filename= categorical_path)
    joblib.dump(value= column_mapping, filename= column_mapping_path)

    # upload
    # #run.upload_file(model_path, model_path)
    # run.upload_file(scaler_path, scaler_path)
    # run.upload_file(categorical_path, categorical_path)
    # run.upload_file(column_mapping_path, column_mapping_path)

    # Registering Model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model( sk_model= model,
                              registered_model_name= args.registered_model_name,
                              artifact_path=  dir_path
                            )

    # Saving the model to a file
    mlflow.sklearn.save_model( sk_model= model,
                               path=os.path.join(args.model, "trained_model"),
                             )    
     

    # Stop Logging
    mlflow.end_run()
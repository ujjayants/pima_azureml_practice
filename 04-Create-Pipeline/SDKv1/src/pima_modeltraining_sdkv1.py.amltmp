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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

if __name__ == '__main__':

    run = Run.get_context()

    # arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="output data from previous step is used as input")
    #parser.add_argument("--train_test_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--output",     type=str, help="output directory")
    args = parser.parse_args()

    # input file path
    input_folder_path = os.path.join(args.input_data)
    print(f"pre-processed data folder path:{input_folder_path}")

    # getting data
    for f in os.listdir(input_folder_path):
        if 'df_processed' in f:
            input_file_path = os.path.join(input_folder_path, f)
            
            # load pickle HERE
            df = pd.read_csv(input_file_path)
            
            print('Processed DataFrame Shape:',df.shape)
            print('Processed DataFrame columns:',df.columns)
            print(df['split_type'].value_counts() )

        # load column mapping
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
    X_train = X_train.drop(columns = ['split_type' ,'Outcome']).reset_index(drop= True)
    X_val = X_val.drop(columns = ['split_type','Outcome']).reset_index(drop= True)

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
    
    X_train = X_train.drop(columns = ['Insulin','BMI']).reset_index(drop= True)

    X_val = X_val.drop(columns = ['Insulin','BMI']).reset_index(drop= True)


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
    
    # without Sampling
    model.fit(X_train_scaled, y_train)
    y_pred= model.predict(X_val_scaled)

    # creating dataframe with actuals and predictions
    df_val =pd.DataFrame({'Actual':y_val, 'Prediction': y_pred}).reset_index(drop= True)
    
    # creating output folder
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created output folder: {args.output}")

    # Saving Data
    save_path1  = os.path.join(args.output , f'df_val.csv')
    save_path2  = os.path.join(args.output , f'pima_pipeline_model.pkl')

    preproc_artifacts_path  = os.path.join(args.output , f'preproc_artifacts.pkl')

    preproc_artifacts = {'scaler':scaler, 'unique_values_train':unique_values_train,'column_mapping':column_mapping}

    # save_path3  = os.path.join(args.output , f'scaler.pkl')
    # save_path4  = os.path.join(args.output , f'unique_values_train.pkl')
    # save_path5  = os.path.join(args.output , f'column_mapping.pkl')

    # maybe dump the model and preprocessing artifacts together here
    df_val.to_csv(save_path1, header=True, index=False)
    joblib.dump( model  ,save_path2 )

    joblib.dump(   preproc_artifacts, preproc_artifacts_path )

    # joblib.dump( scaler  ,save_path3 )
    # joblib.dump(  unique_values_train  ,save_path4 )
    # joblib.dump(  column_mapping  ,save_path5 )

    run.complete()

    

    

    
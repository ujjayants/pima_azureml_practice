# import libraries
#Azure
from azureml.core import Run
#python
import os
import pandas as pd
import numpy as np
import argparse
import warnings
import sklearn
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')


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



if __name__=='__main__':

    #Initializing run object
    run = Run.get_context()

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="pipeline step output directory")
    parser.add_argument("--train_test_ratio", type=float, required=False, default=0.25)
    args = parser.parse_args()

    # Loading input data   
    df = run.input_datasets['raw_data'].to_pandas_dataframe()

    # taking sample of 20% for faster processig
    # df = df.sample(frac=0.2)

    #print(df.columns)

    column_mapping = { df.columns[0]:'Pregnancies',df.columns[1] :'Glucose',
    df.columns[2]:'BloodPressure', df.columns[3] : 'SkinThickness',
    df.columns[4]: 'Insulin' , df.columns[5]:  'BMI',
    df.columns[6] : 'DiabetesPedigreeFunction' , df.columns[7] : 'Age' ,
    df.columns[8] :'Outcome'}

    # Rename columns using the rename() function
    df.rename(columns=column_mapping, inplace=True)

    # Split data

    X_train, X_val, y_train, y_val = train_test_split(df.loc[:, df.columns != 'Outcome'], df['Outcome'], test_size = args.train_test_ratio, stratify= df['Outcome'], random_state = 1234)

    #print(X_train.columns)

    X_train, X_val = impute_mv(X_train, X_val)

    #df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))

    # BMI
    X_train = X_train.assign(BM_DESC= X_train.apply(set_bmi, axis=1))
    X_val = X_val.assign(BM_DESC= X_val.apply(set_bmi, axis=1))

    # Insulin
    X_train = X_train.assign(INSULIN_DESC= X_train.apply(set_insulin, axis=1))
    X_val = X_val.assign(INSULIN_DESC= X_val.apply(set_insulin, axis=1))

    # # Change to type categorical
    # X_train['BM_DESC'] = X_train['BM_DESC'].astype('category')
    # X_val['BM_DESC'] = X_val['BM_DESC'].astype('category')

    # # Change to type categorical
    # X_train['INSULIN_DESC'] = X_train['INSULIN_DESC'].astype('category')
    # X_val['INSULIN_DESC'] = X_val['INSULIN_DESC'].astype('category')

    # # Identify unique categorical values in the training set
    # unique_values_train =  {col: X_train[col].unique() for col in X_train.select_dtypes(include=['category']).columns}

    # for col in X_val.columns:
    #     if col in unique_values_train:
    #         # Ensure that the validation set's categorical column has the same unique values as the training set
    #         X_val[col] = X_val[col].astype('category').cat.set_categories(unique_values_train[col])


    # X_train['split_type'] ='Training'
    # X_val ['split_type'] = 'Validation'

    train_set = pd.concat([X_train,y_train], axis = 1).reset_index(drop = True)
    train_set['split_type'] = 'Training'

    val_set = pd.concat([X_val,y_val], axis = 1).reset_index(drop = True)
    val_set['split_type'] = 'Val'

    df = pd.concat([ train_set, val_set], axis = 0).reset_index(drop =True)

    print(df.shape)
    print(df.columns)
    print(df.split_type.value_counts())

    # saving data
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created pipeline step output folder: {args.output}")

    column_mapping_path = os.path.join(args.output , f'column_mapping.pkl') 
    #f'{args.output}/column_mapping.pkl'
    save_path  = os.path.join(args.output , f'df_processed.csv')

    # save columns mapping and unique train values list

    df.to_csv(save_path, header=True, index=False)
    
    joblib.dump(value= column_mapping, filename= column_mapping_path)

    run.complete()
   
    
     
    

    

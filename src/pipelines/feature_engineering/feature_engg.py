import pandas as pd
from datetime import datetime
import numpy as np 
from logger import logging

def load_data_to_csv_and_feature_engg(account_path,client_path,disp_path,district_path,loan_path):

    # Load CSVs
    loan = pd.read_csv(loan_path)
    account = pd.read_csv(account_path)
    disp = pd.read_csv(disp_path)
    client = pd.read_csv(client_path)
    district = pd.read_csv(district_path)
    logging.info("All required data loaded succesfully")
    
    #  Merge loan with account
    df1 = loan.merge(account, on='account_id', how='left')
    
    # Filter dispositions to owners only and merge client_id
    disp_owner = disp[disp['type'] == 'OWNER']
    df1 = df1.merge(disp_owner[['account_id', 'client_id']], on='account_id', how='left')
    
    # Merge client info (this introduces district_id_x)
    df1 = df1.merge(client[['client_id', 'district_id', 'birth_date','gender']], on='client_id', how='left')
    
    # Rename district_id columns for clarity
    df1.rename(columns={'district_id_x': 'district_id_client', 'district_id_y': 'district_id_account'}, inplace=True)
    
    # Create district mismatch flag
    df1['district_mismatch'] = (df1['district_id_client'] != df1['district_id_account']).astype(int)
    
    # Ensure types match for merge
    df1['district_id_client'] = df1['district_id_client'].astype(str)
    district['district_id'] = district['district_id'].astype(str)
    
    # Merge with district info using client’s district_id
    df1 = df1.merge(district, left_on='district_id_client', right_on='district_id', how='left')

    # Define a reference date for calculating age
    reference_date = datetime(1999, 12, 31)
    
    # Convert birth_date to datetime if not already
    df1['birth_date'] = pd.to_datetime(df1['birth_date'], errors='coerce')
    
    # Derive person age in years
    df1['person_age'] = df1['birth_date'].apply(lambda x: (reference_date - x).days // 365 if pd.notnull(x) else None)

    df1.rename(columns={
    'date_x': 'loan_date',
    'date_y': 'account_open_date'
    }, inplace=True)


    df1['loan_date'] = pd.to_datetime(df1['loan_date'], errors='coerce')
    df1['account_open_date'] = pd.to_datetime(df1['account_open_date'], errors='coerce')
    
    df1['account_age_days'] = (df1['loan_date'] - df1['account_open_date']).dt.days
    df1['account_age_years'] = (df1['account_age_days'] / 365).round(1)

    df1['monthly_burden_ratio'] = df1['payments'] / df1['A11']
    df1['loan_amount_to_income_ratio'] = df1['amount'] / df1['A11']
    
    # Replace invalid values (inf or NaN) with 0 or median
    for col in ['monthly_burden_ratio', 'loan_amount_to_income_ratio']:
        df1[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df1[col].fillna(df1[col].median(), inplace=True)
    logging.info("All feature engineering has been succesfully implemented.")

    df1_good = df1.loc[(df1['status']=='A') | (df1['status']=='C')]
    df1_bad = df1.loc[(df1['status']=='B') | (df1['status']=='D')]

    df2 = df1.copy()

    return df2,df1_good,df1_bad
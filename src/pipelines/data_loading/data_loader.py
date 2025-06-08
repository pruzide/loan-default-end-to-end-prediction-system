import mysql.connector
import pandas as pd
import os
import sys
from logger import logging
from exception import CustomException

def load_data(account_path,card_path,client_path,disp_path,district_path,loan_path,order_path,trans_path):

    try:

        path_list = [account_path,card_path,client_path,disp_path,district_path,loan_path,order_path,trans_path]
    
        # Connection parameters
        conn = mysql.connector.connect(
            host="relational.fel.cvut.cz",
            user="guest",
            password="ctu-relational",
            port=3306,
            database="financial"
        )

        logging.info("Connection established with host")
        
        # List of tables to download
        tables = ['account', 'card', 'client', 'disp', 'district', 'loan', 'order', 'trans']

        # output_folder = r"C:\loan_default_prediction system\data\01_raw"
        # os.makedirs(output_folder,exist_ok=True)

        count = 0

        # Fetch and save each table
        for table in tables:
            cursor = conn.cursor(dictionary=True) # dictionary=True condition makes it easier to store the data as DataFrame 
            cursor.execute(f"SELECT * FROM `{table}`") # ← Backticks fix SQL reserved words"
            rows = cursor.fetchall()
            df = pd.DataFrame(rows)
            
            # output_path = os.path.join(output_folder,f"{table}.csv")
            df.to_csv(path_list[count], index=False)
            print(f"Saved {table}.csv with {len(df)} records.")
            cursor.close()
            count = count + 1
            if (count > len(path_list)):
                break

        logging.info("Data loaded successfully as csv files in data folder")
        conn.close()

    except Exception as e:
        raise CustomException(e,sys)
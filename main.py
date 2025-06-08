### Global Libraries ###

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




### Local Functions and Utilities ###

from logger import logging
from exception import CustomException
from src.pipelines.utils.config_list import configure
from src.pipelines.data_loading.data_loader import load_data
from src.pipelines.feature_engineering.feature_engg import load_data_to_csv_and_feature_engg
from src.pipelines.reporting.data_visualization import visual,sweetviz_visual
from src.pipelines.data_science.encoding import encode,target_encode
from src.pipelines.data_science.modelling import drop_cols,fill_null,split,evaluate_model,final_model

## Getting all CSV file paths from configure file
account_path,card_path,client_path,disp_path,district_path,loan_path,order_path,trans_path,file_path_X_train,file_path_X_test,visual_columns,countplot_columns,eda_path_seaborn,eda_path_sweetviz,feature_cols,cols_to_be_dropped,model_path,shap_plot_path = configure()

## Loading data into our data folder by establishing SQL connection.
load_data(account_path,card_path,client_path,disp_path,district_path,loan_path,order_path,trans_path)

## Engineered Features Final Dataset

df,df_good,df_bad = load_data_to_csv_and_feature_engg(account_path,client_path,disp_path,district_path,loan_path)
df1 = df.copy()

## EDA and Sweetviz reports

visual(df_good,df_bad,visual_columns,countplot_columns,eda_path_seaborn)
sweetviz_visual(df1,eda_path_sweetviz)

## Encoding categorical features and target class

df1 = encode(df1)

df1['frequency_encoded'] = target_encode(df1['frequency'],df1['default'])

## Dropping non-useful columns before modelling phase

df1 = drop_cols(df1,cols_to_be_dropped)

## Filling null values 

df1 = fill_null(df1)

## Splitting data into training and test sets

X_train,X_test,y_train,y_test = split(df1,feature_cols)

## Model Training and Evaluation Metrics along with pickalizing the model and saving the SHAP plot

final_model(model_path,shap_plot_path,file_path_X_train,file_path_X_test,X_train,X_test,y_train,y_test)







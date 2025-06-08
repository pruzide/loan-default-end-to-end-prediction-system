import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from logger import logging
import pickle

def drop_cols(df,cols_to_be_dropped):
    df.drop(labels = cols_to_be_dropped,axis = 1, inplace = True)
    logging.info("Non-useful columns have been dropped.")
    return df


def fill_null(df):
    df[['A12', 'A15']] = df[['A12', 'A15']].fillna(df[['A12', 'A15']].mean())
    logging.info("Null values have been handled.")
    return df


def split(df,feature_cols):
    X=df[feature_cols]
    y=df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3, random_state=42)
    logging.info("train_test_split has been completed.")
    return X_train,X_test,y_train,y_test


def evaluate_model(true,predicted):
    print(classification_report(true,predicted))
    print('\n')
    print('\n')
    print(confusion_matrix(true,predicted))



def final_model(model_path,X_train,X_test,y_train,y_test):

    model = SVC(probability=True,class_weight='balanced', random_state=42,C =1 , gamma='scale', kernel='rbf')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test,y_pred)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Best model has been saved as a pickle file.")
    print(f"Manually selected best model (SVC) saved as {model_path}")
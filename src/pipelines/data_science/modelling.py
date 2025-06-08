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
from sklearn.preprocessing import StandardScaler
from logger import logging
import pickle
import matplotlib.pyplot as plt
import shap


def drop_cols(df,cols_to_be_dropped):
    df.drop(labels = cols_to_be_dropped,axis = 1, inplace = True)
    logging.info("Non-useful columns have been dropped.")
    return df


def fill_null(df):
    df['A15'] = df['A15'].fillna(df['A15'].mean())
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



def final_model(model_path,shap_plot_path,X_train,X_test,y_train,y_test):

    scaler = StandardScaler()

    X_train_df = pd.DataFrame(X_train,columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test,columns=X_test.columns)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("X_train and X_test have been scaled")

    model = SVC(probability=True,class_weight='balanced', random_state=42,C =1 , gamma='scale', kernel='rbf')
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test,y_pred)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Best model has been saved as a pickle file.")
    print(f"Manually selected best model (SVC) saved as {model_path}")

      # SHAP setup
    shap.initjs()

    # Sample data (ensure float64 for SHAP stability)
    background = X_train_df.sample(50, random_state=42).astype(np.float64)
    X_sample = X_test_df.sample(23, random_state=42).astype(np.float64)

    # Store columns
    feature_names = X_train_df.columns.tolist()

    # Fix lambda to wrap NumPy input as a DataFrame
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names)))[:, 1],background)


    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Plot # beeswarm plot
    shap.summary_plot(shap_values, X_sample,show = False)

 
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)

    logging.info("SHAP plot has been saved in reporting folder.")

    plt.close()
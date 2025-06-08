import yaml
import os
from logger import logging

def configure():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processing_path = os.path.abspath(os.path.join(script_dir,'..','..', '..', 'conf', 'base', 'processing.yml'))
    eda_model_path = os.path.abspath(os.path.join(script_dir,'..','..','..', 'conf', 'base', 'eda_modelling.yml'))

    with open(processing_path, 'r') as file:
        processing = yaml.safe_load(file)

    with open(eda_model_path, 'r') as file:
        eda_model = yaml.safe_load(file)


    file_path_account = None
    file_path_card = None
    file_path_client = None
    file_path_disp = None
    file_path_district = None
    file_path_loan = None
    file_path_order = None
    file_path_trans = None

    eda_path_seaborn = None
    eda_path_sweetviz = None
    
    shap_path = None

    model_path = None

    for source in processing.get('account_data', []):
        if source.get('name') == 'account':
            file_path_account = source.get('path')
            logging.info("account.csv path succesfully obtained.")
            break

    for source in processing.get('card_data', []):
        if source.get('name') == 'card':
            file_path_card = source.get('path')
            logging.info("card.csv path succesfully obtained.")
            break

    for source in processing.get('client_data', []):
        if source.get('name') == 'client':
            file_path_client = source.get('path')
            logging.info("client.csv path succesfully obtained.")
            break

    for source in processing.get('disp_data', []):
        if source.get('name') == 'disp':
            file_path_disp = source.get('path')
            logging.info("disp.csv path succesfully obtained.")
            break

    for source in processing.get('district_data', []):
        if source.get('name') == 'district':
            file_path_district = source.get('path')
            logging.info("district.csv path succesfully obtained.")
            break

    for source in processing.get('loan_data', []):
        if source.get('name') == 'loan':
            file_path_loan = source.get('path')
            logging.info("loan.csv path succesfully obtained.")
            break

    for source in processing.get('order_data', []):
        if source.get('name') == 'order':
            file_path_order = source.get('path')
            logging.info("order.csv path succesfully obtained.")
            break

    for source in processing.get('trans_data', []):
        if source.get('name') == 'trans':
            file_path_trans = source.get('path')
            logging.info("trans.csv path succesfully obtained.")
            break


    visual_columns = eda_model.get('visual_columns',[])
    countplot_columns = eda_model.get('countplot_columns',[])
    feature_columns = eda_model.get('feature_cols',[])
    columns_to_be_dropped = eda_model.get('cols_to_be_dropped',[])



    for source in eda_model.get('seaborn_plots', []):
        if source.get('name') == 'seaborn':
            eda_path_seaborn = source.get('path')
            break

    for source in eda_model.get('sweetviz_html_visualization', []):
        if source.get('name') == 'sweetviz':
            eda_path_sweetviz = source.get('path')
            break

    for source in eda_model.get('best_model', []):
        if source.get('name') == 'best_model':
            model_path = source.get('path')
            break

    for source in eda_model.get('shap_plots', []):
        if source.get('name') == 'shap':
            shap_path = source.get('path')
            break

    return file_path_account,file_path_card,file_path_client,file_path_disp,file_path_district ,file_path_loan,file_path_order,file_path_trans,visual_columns,countplot_columns,eda_path_seaborn,eda_path_sweetviz,feature_columns,columns_to_be_dropped,model_path,shap_path

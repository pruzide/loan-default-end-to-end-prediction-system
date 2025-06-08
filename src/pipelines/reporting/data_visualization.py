from logger import logging

def visual(df1_good,df1_bad,visual_cols,countplot_cols,eda_path_seaborn):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use('default') 

    with PdfPages(eda_path_seaborn) as pdf:
        for col in visual_cols:
            fig , ax = plt.subplots(figsize = (20,10))
            if col not in countplot_cols:
                plt.figure(figsize=(12,6))
                sns.histplot(x=str(col), data = df1_good,label = 'good',color='blue',ax=ax)
                sns.histplot(x=str(col), data = df1_bad, color = 'red',label = 'bad',ax=ax)
            else:
                plt.figure(figsize=(20,10))
                sns.countplot(x=str(col), data = df1_good,label = 'good',color = 'blue',ax=ax)
                sns.countplot(x=str(col), data = df1_bad, color = 'red',label = 'bad',ax=ax)
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)  
            plt.close(fig)    

    logging.info("Seaborn plots report generated.")



def sweetviz_visual(df,path):
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html(path,open_browser=False)
    logging.info("Sweetviz report generated.")
            
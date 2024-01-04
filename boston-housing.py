import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os 

def getdata(filepath): #读取sogou的数据
    sep_sign = ","
    with open(file_path, 'r') as file:
        lines = file.readlines()
        comma_count = sum([line.count(',') for line in lines[:5]])
        semicolon_count = sum([line.count(';') for line in lines[:5]])
        
    if comma_count > semicolon_count:
        sep_sign  = ","
    else:
        sep_sign  = ";"
    df = pd.read_csv(filepath ,  sep =  sep_sign)
    df = df.select_dtypes(exclude=['object']) #去掉object类型的数据
    df = df.fillna(df.mean())
    
    return df

filepaths = [ "./data/HousingData.csv" ,  "./data/wine+quality/winequality-red.csv",  "./data/wine+quality/winequality-white.csv" , "./data/online+news+popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv" ]
if not os.path.exists("./pictures"):
    os.mkdir("./pictures")


for file_path in filepaths:
    cur_name = (file_path.split("/")[-1]).split(".")[0]
    if not os.path.exists("./pictures/" +cur_name):
        os.mkdir("./pictures/" +cur_name)
    pic_path = "./pictures/" +cur_name
    df = getdata(file_path)
    
    title_name = df.columns.tolist()
    select_name = title_name[::3]
    data = df.to_numpy()
    for name in select_name:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[name], bins=30, kde=True, color='red')
        plt.title('Distribution of {}' .format(name))
        plt.xlabel('{} ' .format(name))
        plt.ylabel('Frequency')
        cur_save_path = os.path.join(pic_path , "{}-distribution.png" .format(name))
        plt.savefig(cur_save_path)
        plt.close()
    fig_len = 2*len(select_name)
    plt.figure(figsize=(fig_len, fig_len))
    sns.pairplot(df[select_name])
    plt.suptitle('Pairplot of Selected Variables in {}' .format(cur_name), y=1.02)
    cur_save_path = os.path.join(pic_path , "{}-pair.png" .format(cur_name))
    plt.savefig(cur_save_path, dpi= 100)
    plt.close()
    plt.figure(figsize=(fig_len, fig_len))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=(.1*len(select_name)))
    plt.title('Correlation Matrix Heatmap in {}' .format(cur_name))
    cur_save_path = os.path.join(pic_path , "{}-corr.png" .format(cur_name))
    plt.savefig(cur_save_path , dpi= 100)
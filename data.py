import pandas as pd
import numpy as np

def dataget():
    train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')  # 读入数据
    x_train = train_data.iloc[:, 1:80]
    y_train = train_data.iloc[:, 80:]
    x_train_numeric = x_train.select_dtypes(include=[np.number])
    x_train = x_train_numeric.values
    y_train = y_train.values
    return x_train, y_train

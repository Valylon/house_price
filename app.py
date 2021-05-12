import io

import flask
from flask import jsonify, make_response, render_template

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)y
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

app = flask.Flask(__name__, static_url_path='')


def toText(s):
    response = make_response(s, 200)
    response.mimetype = "text/plain"
    return response



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


@app.route('/step3', methods=['GET'])
def step3():
    return str(train_data.shape) + '\n' + str(test_data.shape)


@app.route('/step4-5', methods=['GET'])
def step4_5():
    return train_data.head().to_html() + '<br><br>' + test_data.head().to_html()



@app.route('/step11', methods=['GET'])
def step11():
    train_data = pd.read_csv('train.csv')

    # Filling LotFrontage of training dataset
    train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())

    # Droping Alley column because it has too much null values
    train_data.drop(['Alley'], inplace=True, axis=1)

    # Filling BsmtQual of training dataset
    train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])

    # Filling BsmtCond of training dataset
    train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])

    # Filling FireplaceQu,GarageType,GarageFinish,GarageQual,GarageCond
    train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])
    train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
    train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
    train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
    train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])

    # droping GarageYrBlt,PoolQC,Fence,MiscFeature,ID
    train_data.drop(['GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
    train_data.drop(['Id'], inplace=True, axis=1)

    return str(train_data.shape)

@app.route('/step15-17', methods=['GET'])
def step15_17():
    train_data = pd.read_csv('train.csv')

    step11()

    train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())

    # Droping Alley column because it has too much null values
    train_data.drop(['Alley'], inplace=True, axis=1)

    # Filling BsmtQual of training dataset
    train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])

    # Filling BsmtCond of training dataset
    train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])

    # Filling FireplaceQu,GarageType,GarageFinish,GarageQual,GarageCond
    train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])
    train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
    train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
    train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
    train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])

    # droping GarageYrBlt,PoolQC,Fence,MiscFeature,ID
    train_data.drop(['GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
    train_data.drop(['Id'], inplace=True, axis=1)
    #######

    train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])

    train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])

    train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])

    train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
    train_data.dropna(inplace=True)

    return str(train_data.shape)


@app.route('/step18', methods=['GET'])
def step18():
    return str(test_data.shape)


@app.route('/step19-20', methods=['GET'])
def step19_20():
    test_data = pd.read_csv('test.csv')
    for col in test_data:
        if test_data[col].isna().sum():
            if test_data[col].dtype == 'O':
                test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
            else:
                test_data[col] = test_data[col].fillna(test_data[col].mean())

    test_data.drop(['GarageYrBlt'], axis=1, inplace=True)
    test_data.drop(['Id'], axis=1, inplace=True)
    test_data.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
    test_data.drop(['Alley'], axis=1, inplace=True)
    return str(test_data.shape)


@app.route('/step21-22', methods=['GET'])
def step21_22():
    step15_17()
    step19_20()
    return '<h3>Train data</h3>' + train_data.head().to_html() + '<br><br><h3>Test data</h3>' + test_data.head().to_html()



@app.route('/step23-24', methods=['GET'])
def step23_24():
    step21_22()
    global categorial_col
    categorial_col = [col for col in train_data if train_data[col].dtype == 'O']
    return str(categorial_col)


@app.route('/step25', methods=['GET'])
def step25():

    step23_24()
    return train_data.head().to_html()

@app.route('/step26-27', methods=['GET'])
def step26_27():
    step25()
    final_df = pd.concat([train_data, test_data], axis=0)
    return toText(final_df.isnull().sum().to_string())


@app.route('/step30-37', methods=['GET'])
def step30_37():
    step25()

    def category_onehot_multcols(multcolumns):
        df_final = final_df
        i = 0
        for fields in multcolumns:

            # print(fields)
            df1 = pd.get_dummies(final_df[fields], drop_first=True)

            final_df.drop([fields], axis=1, inplace=True)
            if i == 0:
                df_final = df1.copy()
            else:

                df_final = pd.concat([df_final, df1], axis=1)
            i = i + 1

        df_final = pd.concat([final_df, df_final], axis=1)

        return df_final

    global final_df
    final_df = pd.concat([train_data, test_data], axis=0)
    final_df.isnull().sum()
    # final_df['SalePrice']
    # final_df.shape
    final_df = category_onehot_multcols(categorial_col)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    return final_df.to_html()




@app.route('/step38-40', methods=['GET'])
def step38_40():
    step30_37()
    global df_Train
    global df_Test
    df_Train = final_df.iloc[:1422, :]
    df_Test = final_df.iloc[1422:, :]

    return '<h3>Train df </h3>' + df_Train.head().to_html() + '<br><br><h3>Test df</h3>' + df_Test.head().to_html()

@app.route('/step41-44', methods=['GET'])
def step41_44():
    step38_40()
    df_Test = final_df.iloc[1422:, :]
    df_Train = final_df.iloc[:1422, :]

    df_Test.drop(['SalePrice'], axis=1, inplace=True)
    global X_train
    global y_train
    X_train = df_Train.drop(['SalePrice'], axis=1)
    y_train = df_Train['SalePrice']

    classifier = xgboost.XGBRegressor()
    classifier.fit(X_train, y_train)
    global y_pred
    y_pred = classifier.predict(df_Test)

    return str(y_pred)


@app.route('/step47-54', methods=['GET'])
def step47_54():
    step41_44()
    pred = pd.DataFrame(y_pred)
    sub_df = pd.read_csv('sample_submission.csv')
    datasets = pd.concat([sub_df['Id'], pred], axis=1)
    datasets.columns = ['Id', 'SalePrice']
    datasets.to_csv('final_submission.csv', index=False)
    fdf = pd.read_csv('final_submission.csv')
    convert_dict = {'Id': int,
                    'SalePrice': float
                    }
    for col in fdf:
        if fdf[col].isna().sum():
            pass
            # print(col)

    len(fdf)
    is_NaN = fdf.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = fdf[row_has_NaN]
    # print(rows_with_NaN)

    subdf = pd.read_csv('sample_submission.csv')

    buf = io.StringIO()
    subdf.info(buf=buf)
    s = subdf.head().to_string()

    return toText(s)

@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=False)

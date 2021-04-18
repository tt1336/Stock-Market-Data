from pathlib import Path

import bs4 as bs
from collections import Counter
import datetime as dataframe
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
from sklearn import svm, neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

x = ("AMZN")

def get_data(i):
   start = dataframe.datetime(2019, 6, 8)
   end = dataframe.datetime.now()
   df = web.get_data_yahoo(i, start, end)
   df.reset_index(inplace=True)
   df.set_index("Date", inplace=True)
   df.to_csv('stock_database/{}.csv'.format(i))

get_data(x)
def visualization(i):
   df = pd.read_csv("stock_database/{}.csv".format(i))
   pd.set_option("display.precision",3)
   df["Low"].plot()
   plt.show()
   pd.set_option('max_rows', 99999)
   pd.set_option('max_colwidth', 400)
   pd.describe_option('max_colwidth')


visualization(x)
def establishing_list():
    test = []
    for subdir, dirs, files in os.walk("stock_database"):
        for file in files:
            temp = file.replace(".csv","")
            test.append(temp)

    df = pd.DataFrame(test)
    df.to_csv('file2.csv', index=False, header=False)
    return test

establishing_list()

def corrlation():
    central = pd.DataFrame()
    for x in establishing_list():
        df = pd.read_csv('stock_database/{}.csv'.format(x))
        df.set_index('Date', inplace=True)

        df.rename(columns={'High': x}, inplace=True)
        df.drop(['Open', 'Adj Close', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if central.empty:
            central = df
        else:
            central = central.join(df, how='outer')


    central.to_csv('combinedstocks.csv')


corrlation()

def corrlation_graph():
    df = pd.read_csv('combinedstocks.csv')
    df_corr = df.corr()

    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

corrlation_graph()
def graphs(x):
    df = pd.read_csv('stock_database/{}.csv'.format(x))
    median = df
    df.plot(kind='hexbin', x='High', y='Low', gridsize=25)
    plt.show()
    df.plot(kind='hist')
    plt.show()

graphs(x)

def precent_change_data(stock):
   days=5
   df = pd.read_csv("combinedstocks.csv", index_col=0)
   tickers = df.columns.values.tolist()
   df.fillna(0, inplace=True)

   for i in range(1, days + 1):
       df['{}_{}d'.format(stock, i)] = (df[stock].shift(-i) - df[stock]) / df[stock]

   df.fillna(0, inplace=True)
   return tickers, df, days


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def data_gathering(ticker):
    tickers, df, days = precent_change_data(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)] for i in range(1, days+1)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df

def do_ml(ticker):
    X, y, df = data_gathering(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    clfTwo = LinearRegression().fit(X,y)
    clfTwo.fit(X_train,y_train)

    clfThree = RandomForestClassifier(max_depth=3,random_state=0)
    clfThree.fit(X_train,y_train)

    confidenceofKNeighbor = clf.score(X_test, y_test)
    confidenceofLinear = clfTwo.score(X_test, y_test)
    confidenceofForest = clfThree.score(X_test, y_test)

    predictions = clf.predict(X_test)
    predictionsTwo = clfTwo.predict(X_test)
    predictionsThree = clfThree.predict(X_test)


    print('accuracy of kNeigbor:', confidenceofKNeighbor)
    print('predicted class counts:', Counter(predictions))
    print('linear :', confidenceofLinear)
    print('predicted class counts:', Counter(predictionsTwo))
    print('accuracy of forest :', confidenceofForest)
    print('predicted class counts:', Counter(predictionsThree))

    print()
    print()
    return confidenceofKNeighbor

do_ml(x)
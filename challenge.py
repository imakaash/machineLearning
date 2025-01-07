# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 11:43:48 2023

@author: akash
"""

def run():
    import numpy as np
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    warnings.filterwarnings('ignore')
    
    random_state = 42
    np.random.seed(random_state)
    
    print('''DATASET AND TASK\n''')
    
    train_data = pd.read_csv(r'cars_train.csv')
    test_data = pd.read_csv(r'cars_test_without_labels.csv')
    print('Train data info:')
    print(train_data.info(),'\n')
    
    print('Train data head:')
    print(train_data.head(),'\n')

    train_data[['seats', 'previous_owner', 'cylinder', 'gears']] = train_data[['seats', 'previous_owner', 'cylinder', 'gears']].astype(str)
    test_data[['seats', 'previous_owner', 'cylinder', 'gears']] = test_data[['seats', 'previous_owner', 'cylinder', 'gears']].astype(str)
    
    train_data = train_data.iloc[:,1:]
    test_id = test_data['ID']
    test_data = test_data.iloc[:,1:]
    
    print('Describing the numerical features before imputing:')
    print(train_data.describe(),'\n')
    
    def impute_outliers_IQR(df):
       q1 = df.quantile(0.25)
       q3 = df.quantile(0.75)
    
       IQR = q3-q1
    
       upper = df[~(df>(q3+1.5*IQR))].max()
       lower = df[~(df<(q1-1.5*IQR))].min()
    
       df = np.where(df > upper, np.median(df), np.where(df < lower, np.median(df), df))
       return df
    
    train_data['year'] = impute_outliers_IQR(train_data['year'])
    test_data['year'] = impute_outliers_IQR(test_data['year'])
    
    train_data['cubic_capacity'] = impute_outliers_IQR(train_data['cubic_capacity'])
    test_data['cubic_capacity'] = impute_outliers_IQR(test_data['cubic_capacity'])
    
    train_data['kilometers'] = impute_outliers_IQR(train_data['kilometers'])
    test_data['kilometers'] = impute_outliers_IQR(test_data['kilometers'])
    
    # def impute_outliers_Trim(df):
    #    q1 = df.quantile(0.25)
    #    q3 = df.quantile(0.75)
    
    #    IQR = q3-q1
    
    #    upper = df[~(df>(q3+1.5*IQR))].max()
    #    lower = df[~(df<(q1-1.5*IQR))].min()
    
    #    df = np.where(df > upper, df.quantile(0.99), np.where(df < lower, df.quantile(0.01), df))
    #    return df
    
    # train_data['year'] = impute_outliers_Trim(train_data['year'])
    # test_data['year'] = impute_outliers_Trim(test_data['year'])
    
    # train_data['cubic_capacity'] = impute_outliers_Trim(train_data['cubic_capacity'])
    # test_data['cubic_capacity'] = impute_outliers_Trim(test_data['cubic_capacity'])
    
    # train_data['kilometers'] = impute_outliers_Trim(train_data['kilometers'])
    # test_data['kilometers'] = impute_outliers_Trim(test_data['kilometers'])
    
    print('Describing the numerical features after imputing:')
    print(train_data.describe(),'\n')
    
    var = 'fuel'
    data = pd.concat([train_data['price'], train_data[var]], axis=1)
    f, ax = plt.subplots(figsize=(20, 12))
    fig = sns.boxplot(x=var, y="price", data=data)
    plt.show()
    
    var = 'year'
    data = pd.concat([train_data['price'], train_data[var]], axis=1)
    f, ax = plt.subplots(figsize=(20, 12))
    fig = sns.boxplot(x=var, y="price", data=data)
    plt.xticks(rotation=90)
    plt.show()
    
    print('Number of unique values in each Categorical feature:')
    print(train_data.select_dtypes(include='object').nunique(), '\n')
    
    # sig = train_data.model.value_counts(normalize=True).to_frame()
    # model_lst = sig[sig.model>0.009].index.to_list()
    
    # train_data.model = train_data.model.apply(lambda x: x if x in model_lst else 'other')
    # test_data.model = test_data.model.apply(lambda x: x if x in model_lst else 'other')

    # dropping model attribute as it has lot of unique values and is not important
    train_data.drop(["model"],axis=1,inplace=True)
    test_data.drop(["model"],axis=1,inplace=True)
    
    for col in train_data.select_dtypes(include='object').columns:
      train_data[col] = train_data[col].apply(lambda x : x.lower())
      test_data[col] = test_data[col].apply(lambda x : x.lower())
      
    for col in train_data.select_dtypes(include='object').columns:  
      var = col
      f, ax = plt.subplots(figsize=(10, 10))
      fig = sns.countplot(x=train_data[var])
      plt.xticks(rotation=90)
      plt.xlabel(col)
      plt.show()
      
    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)
    
    print('Shape of data after pre-processing:')
    print('-----------------------------------')
    print("Shape of train data: {}\nShape of test data: {}\n".format(train_data.shape, test_data.shape))
    
    train_columns = list(train_data.columns)
    train_columns.remove('price')
    missing_columns = set(train_columns).difference(test_data.columns)
    print('Missing columns in test dataset: ',missing_columns, '\n')
    
    for col in missing_columns:
      test_data[col] = [0]*len(test_data)
      
    missing_columns = set(train_columns).difference(test_data.columns)
    print('Missing columns in test dataset after fix: ',missing_columns, '\n')
    
    print("Shape of train data: {}\nShape of test data: {}\n".format(train_data.shape, test_data.shape))
    
    from sklearn.preprocessing import StandardScaler
    train_data_scaled = train_data[train_columns].values
    train_data_scaled = np.asarray(train_data_scaled)
    
    test_data_scaled = test_data[train_columns].values
    test_data_scaled = np.asarray(test_data_scaled)
    
    # Finding normalised array of X_Train
    train_data_scaled = StandardScaler().fit_transform(train_data_scaled)
    test_data_scaled = StandardScaler().fit_transform(test_data_scaled)
    
    print('Splitting data for training and evaluating the results:')
    print('------------------------------------------------------')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_data_scaled, train_data['price'], test_size=0.2, random_state = random_state)
    print("Shape of train features: {}\nShape of test features: {}\nShape of train labels: {}\nShape of test labels: {}\n".format(X_train.shape, X_test.shape, len(y_train), len(y_test)))
    
    print('Model Building:')
    print('--------------')
    print('\n1.LINEAR REGRESSION:')
    from sklearn import metrics
    from sklearn import linear_model
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    print(linear_reg, '\n')
    print("Accuracy on Traing set: ",linear_reg.score(X_train,y_train))
    print("Accuracy on Testing set: ",linear_reg.score(X_test,y_test))
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))    
    
    
    print('\n2.RIDGE REGRESSION:')
    ridge_model = linear_model.Ridge()
    ridge_model.fit(X_train,y_train)
    y_pred = ridge_model.predict(X_test)
    print(ridge_model, '\n')
    print("Accuracy on Traing set: ",ridge_model.score(X_train,y_train))
    print("Accuracy on Testing set: ",ridge_model.score(X_test,y_test))
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
    
    
    print('\n3.LASSO REGRESSION:')
    lasso_model = linear_model.Lasso()
    lasso_model.fit(X_train,y_train)
    y_pred = lasso_model.predict(X_test)
    print(lasso_model, '\n')
    print("Accuracy on Traing set: ",lasso_model.score(X_train,y_train))
    print("Accuracy on Testing set: ",lasso_model.score(X_test,y_test))
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))
    
    y_pred = lasso_model.predict(test_data_scaled)
    d = {'ID': test_id, 'predicted_price':y_pred}
    output = pd.DataFrame(d)
    output['predicted_price'] = output['predicted_price'].apply(lambda x: round(x)) 
    output.to_csv(r'predictions_Yadav_Akash.csv', index=None)
    
if __name__ == '__main__':
    run()
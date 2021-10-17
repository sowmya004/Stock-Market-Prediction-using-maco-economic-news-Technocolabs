# PREDICTING THE VOLATILITY IN EQUITY MARKETS USING MACRO ECONOMIC HEADLINES

A) INTRODUCTON:

On June 2015, 2016 debt negotiations between Greek Govt and its creditors borke off abrubptly. Large market movements as a concequence of political and economic headlines are hardly uncommon, liquid markets are most suspectable to swing when the news breaks. Using VIX as a proxy for market volatality, we investigate how macroeconoic headlines affect the changes. Here, we predict equity market value using tweets from major news sources, investment banks and notable economists.

B) PROBLEM STATEMENT:

Twitter provides a plethora of market data. In this project we have extracted around 100,000 tweets from various accounts to predict the upward movements. Using this data we are researching how this economic news affects the market.

C) TYPE OF MACHINE LEARNING:

This project is Regression based problem, which is a predictive modelling technique that analyzes the relation between the target or dependent variable and independent variable in a dataset.

METRICS USED: The performance of a regression model must be reported as an error in those predictions and these error summarizes on average how close the predictions were to their expected values.

Accuracy mectrics we have used in this project are:

1.)Root Mean Squared Error(RMSE)
2.)Mean Absolute Error(MAE)
3.)Rsquared value(r2)

## EXPLORATORY DATA ANALYSIS:

EDA includes extracting the twitter data based on the stock names viz, Apple, Tesla, Nvidia, Paypal and Microsoft, cleaning of twitter data that were pulled i.e., removing unnecessary data from tweets. After cleaning the data, below are the plots that were plotted against the sentiments that is Positive, Negative and Neutral.

![image](https://user-images.githubusercontent.com/72294006/137591119-28660a88-8d83-4501-8c8c-76102dc86556.png)

![image](https://user-images.githubusercontent.com/72294006/137591124-fe48062b-19b1-4b50-8198-9d77f36e9978.png)

![image](https://user-images.githubusercontent.com/72294006/137591135-caececd4-951b-4cf7-aba9-ca9b92c934b9.png)

![image](https://user-images.githubusercontent.com/72294006/137591144-88d9741a-446e-45cb-aa19-19f6137f8475.png)


## MODELLING:

We have implemented differnt ML models Linear Regression, Random Forest Regression, Decision Tree Regressor. We  chosed Linear Regression ML for our project as its r2 - 0.99974, rmse - 2.65. Below are the plots which supports our decision

![image](https://user-images.githubusercontent.com/72294006/137591748-b6915b5b-5c11-4cfa-aed5-30a50acc7ee4.png)


  x  = df5[['Year','Month','Day','StockName','Positive','Negative','Neutral']].to_numpy()

  y = np.array(df5['Close'])

   for train_index, test_index in tscv.split(x):

    x_train , x_test = x[train_index] , x[test_index]
    
    y_train , y_test = y[train_index] , y[test_index]
    
    regresor = LinearRegression()
    
    regresor.fit(x_train,y_train)
    
    y_pred = regresor.predict(x_test)
    
    rmse = (math.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    
    mae = (metrics.mean_absolute_error(y_test, y_pred))
    
    r2 = metrics.r2_score(y_test, y_pred)



## GRAPH:



![image](https://user-images.githubusercontent.com/72294006/137591674-fee9fe96-8b08-4c60-82d4-320bdeedcb32.png)

## RELATION BETWEEN FEATURES:


![image](https://user-images.githubusercontent.com/72294006/137591689-469a548b-dd96-4661-bffd-603a2b773e67.png)

![image](https://user-images.githubusercontent.com/72294006/137591691-01e3999d-50f2-486b-9c8a-12badc3aab33.png)

## DEPLOYMENT:

We have deployed the model using Flask framework, as it is a opensource Python library that allows us to create beautiful web apps for Machine Learning. It is hosted on Heroku, as it a container based Platform As A Service(PAAS), because it is flexible and easy to host on this platform.

Heroku : [Visit here](https://stock-prediction-news-headline.herokuapp.com/)

Video  : [Click here](https://drive.google.com/file/d/1xI_m45h5ejmMf9CN2_SRyZQku9IqhahU/view?usp=sharing)








## TEAM MEMBERS

Sowmya Prakash


Anirudh Saxena




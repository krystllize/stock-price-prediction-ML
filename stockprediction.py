import pandas_datareader as pdr
from datetime import date
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sbn; sbn.set()

# Get ticker data from yahoo up to today's date
ticker = input("What stock ticker would you like to analyze?")
raw_data = pdr.data.DataReader(str(ticker), 'yahoo', start='2000-01-01', end=str(date.today().strftime('%Y-%m-%d')))

# Days prediction
days = input("How many days would you like to predict into the future?")


# ############################################ SVM Prediction ##########################################################

def prepData(prediction_length):
    data = raw_data.filter(['Close'])  # Get data with only 'Close' column, and arrange it into an array
    data['Predicted Prices'] = data[['Close']].shift(-prediction_length) # Create prediction time length and lag the
    # column by the value of the prediction length days

    # Create variable X where we will store our dataset to train the data
    x_var = np.array(data.drop(['Predicted Prices'], axis=1))
    x_var = x_var[:-prediction_length]  # Shows data and removing the last prediction_length rows from the data

    y_var = np.array(data['Predicted Prices'])
    y_var = y_var[:-prediction_length]  # dependent variable, column where we will have our future price predictions,
    # and we get all of our y values except for the last [prediction_length] days

    # 80% training 20% testing split
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2)

    # Here, rbf would be the most appropriate kernel since we want more accuracy and generally outperforms other
    # kernels (linear, polynomial, etc.). After doing a bit of research, the most common parameters when using an rbf
    # kernel were C = 1e3 and gamma = 0.1. For other kernels, such parameters differed slightly, but the C
    # regularization parameter tended to be 1e3. This is a relatively large value, which means we're going to be
    # choosing a smaller hyperplane, and with a low gamma value, we take into consideration more points that are
    # located further away from the plane. Therefore, the parameters for the support vector regression depend on
    # which kernel you select.
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)  # fit the regression model

    # testing to see if correlation coefficient fits well, see if our predictions were close to the actual price
    svm_score = svr_rbf.score(x_test, y_test)
    score = round(svm_score, 3)
    print(" ")
    print("svm confidence (correlation coefficient): " + str(svm_score))

    forecast_prices = np.array(data.drop(['Predicted Prices'], 1))[-prediction_length:]  # setting the forecast prices
    # to the last [prediction_length] rows in the 'Close' column

    svm_prediction = svr_rbf.predict(forecast_prices) # run the svm prediction model and spit out the predicted values
    print(" ")
    print('svm prediction values: ', svm_prediction)

    future_dates = pd.date_range(date.today(), periods=prediction_length).tolist()  # getting future dates from today
    # to plot against our predicted prices

    # plot figure details
    plt.figure(figsize=(18, 10))
    plt.style.use('seaborn-darkgrid')
    plt.grid(which='major', linestyle='-', color='black', axis='y')
    plt.grid(which='major', linestyle='-', color='black', axis='x')
    plt.minorticks_on()
    plt.grid(which='minor', color='#d3d3d3', linestyle='-', axis='y')
    plt.grid(which='minor', color='#d3d3d3', linestyle='-', axis='x')
    plt.plot(future_dates, svm_prediction, c='black', label='RBF prediction')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price ($USD)')
    plt.title('Predicted Future Stock Price of ' + str(ticker) + ' for the next ' + str(days) + ' days using Support '
                                                                                                'Vector Regression '
                                                                                                'with a ' + str(
        score) + ' confidence score / correlation coefficient')
    plt.legend()
    plt.show()

    # ############################## Logistic Regression Prediction ####################################################

    data2 = raw_data.dropna()  # drop any rows containing NaN values
    data2 = data2.iloc[:,
            :4]  # get the open column, High price column, low price column, and close column alongside the date
    open = data2['Open']  # isolate the open column from the dataset
    close = data2['Close']  # isolate the close column from the dataset

    # We can use a variety of technical indicators that will help us better predict the movement of the stock price.
    # Here, we will be using a moving average and a rolling correlation to accompany it.

    data2['S_5'] = close.rolling(window=5).mean()  # moving average 5 days
    data2['Corr'] = close.rolling(window=5).corr(data2['S_5'])  # calculate correlation between the 5 moving day
    # average and the closing values for each row

    # We want to subtract the opening price of a chosen day by the closing price of the previous day to account for the
    # potential movement differences over the course of days.
    data2['Open-Close'] = open - close.shift(1)
    data2['Open-Open'] = open - open.shift(1)
    data2 = data2.dropna()
    x = data2.iloc[:, :9]  # independent variable

    y = np.where(close.shift(-1) > close, 1, -1)  # we create a new variable to account for the situation if the closing
    # price is higher than the close of the actual day, assigning the parameter as 1 if true, and -1 if false

    split = int(np.ceil(0.8 * len(data2)))  # split the data into 80% train and 20% test
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]

    logregression = LogisticRegression()  # logistic regression model
    logregression = logregression.fit(x_train, y_train)

    # This is where we start using probabilities to see how likely the stock will move up or down on each day. The
    # probabilities will be displayed under "0" and "1" headers, where 0 corresponds to the probability that the stock
    # will go down, and 1 is the probability that it will go up. We will essentially be constructing a table where it
    # will display the probabilities of movement for each day.
    probability = logregression.predict_proba(
        x_test)  # probability that our model's outcome will go up or down each day
    probability_value = logregression.predict(
        x_test)  # uses the calculated probability and assigns the value "1" if the
    # overall probability says the stock will go up, and assigns "-1" if it predicts it will go down.
    probability_table = pd.DataFrame(probability, columns=['0', '1'])
    probability_value_column = pd.DataFrame(probability_value, columns=['Prediction'])
    table = pd.concat([probability_table, probability_value_column], axis=1,
                      sort=False)  # allows us to concatenate the two columns together since we are working under pandas
    table['Up/Down'] = np.where(table['Prediction'] == 1, 'Up',
                                'Down')  # this allows us to directly label each day as "Up" or "Down" based on if our
    # "Up or Down Prediction" column shows a 1 or a -1: if shows a 1, it will display "Up"
    add_date = x_test.index  # add on date column on the left
    table.set_index([add_date])
    print(" ")
    print("logistic regression synthesis table: ")
    print(table)

    data2['Prediction Signal'] = logregression.predict(x) # predict movements using log regression
    data2['Original Returns'] = np.log(data2['Close'] / data2['Close'].shift(1)) # adjusting position for the
    # original returns of the stock to compare against the predicted returns
    cumulative_original_data = np.cumsum(data2[split:]['Original Returns'])
    data2['Predicted Returns'] = data2['Original Returns'] * data2['Prediction Signal'].shift(1) # adjusting position
    # for the predicted values
    cumulative_predicted_returns = np.cumsum(data2[split:]['Predicted Returns'])

    graph_data = pd.concat([cumulative_original_data, cumulative_predicted_returns], axis=1)
    plt.figure(figsize=(18, 10))
    plt.title('Original vs. Predicted Price of ' + str(ticker) + ' for days leading up to today')
    plt.ylabel('Returns')
    sbn.lineplot(data=graph_data)
    plt.show()

    return


prepData(int(days))  # Running function

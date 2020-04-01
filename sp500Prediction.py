from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Get data
ticker = "SPY"
my_key = "your-API-key"
ts = TimeSeries(key=my_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')

# Create dataframe, drop null values
df = pd.DataFrame(data)
df.fillna(value=-99999, inplace=True)
forecast_col = '4. close'
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())

#Set X and y values
X = np.array(df.drop(columns = ['label'], axis = 1))
X = preprocessing.scale(X)

#What we will forecast against
X_lately = X[-forecast_out:]

#The rest of the X data, used for the regression model and confidence
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Regression
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

#Confidence
confidence = clf.score(X_test, y_test)

#Forecasting
forecast_set = clf.predict(X_lately)

#Results
print(forecast_set, confidence, forecast_out)

#Add forecast column
df['Forecast'] = np.nan

#Get last time in dataframe so we can assign a new time to each forecast
last_time = df.iloc[-1].name
last_unix = last_time.timestamp()
one_minute = 60
next_unix = last_unix + one_minute

#Set next row values in forecast dataframe
for i in forecast_set:
    next_time = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 60
    df.loc[next_time] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#Graph
df['4. close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
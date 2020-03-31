from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Get data
ticker = "SPY"
my_key = "your API key"
ts = TimeSeries(key=my_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')

#Graph
data['4. close'].plot()
plt.title('Intraday Times Series for '+ ticker +'(1 min)')
plt.show()

# Create dataframe, drop null values, set X and y values
df = pd.DataFrame(data)
df.dropna(how='any', inplace=True)
X = df.drop(columns = ['4. close'], axis = 1)
y = df['4. close']

#Regression
clf = linear_model.LinearRegression()
clf.fit(X, y)
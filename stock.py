import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas_datareader as pdr
from pandas_datareader import data, wb
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
dataset=pd.read_csv('C:/Users/Akhilesh Jadhav/Desktop/SPY.csv')

#Setting the Star and Stop Date And Extracting data From Yahoo
start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')
start_date_2 = pd.to_datetime('2015-12-18')
spy = pdr.data.get_data_yahoo('SPY', start_date, stop_date)
xyz = pdr.data.get_data_yahoo('SPY', start_date_2, stop_date)
#print(spy)
#Declaring some variable according to our convenince to plot first graph
spy_c = spy['Close']
spy_high = spy['High']
spy_vol = spy['Volume']/(10E6)
print("\n\n\n")
print("---------------------------------------------------------------------------------------------------")
#Analyzing the data at basic level
spy_first_open = spy['Open'].iloc[0]
spy_last_close = spy['Close'].iloc[-1]
profit_total_period = spy_last_close-spy_first_open
print("The profit if you had invested in first day and sold the stock on last date = ",profit_total_period)
spy['Daily Change'] = pd.Series(spy['Close']-spy['Open'])
daily_change = spy['Daily Change'].sum()
print("If you had bought and sold stocks daily on Open and Close rate, the profit = ",daily_change)
if (profit_total_period>daily_change):
    max_profit=profit_total_period
    t = "Total Period"
else:
    max_profit=daily_change
    t= "Daily Basis"
print("The maximum profit is in",t," =",max_profit)
print("---------------------------------------------------------------------------------------------------")
print("\n\n\n")




df=dataset.sample(800)
low_time = datetime(2008, 12, 1)
low_px = 70

high_time = datetime(2021, 1, 1)
high_px = 370
annotations = []
annotations.append(go.layout.Annotation(x=low_time,
                                        y=low_px,
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowcolor="purple",
                                        arrowsize=2,
                                        arrowwidth=2,
                                        text="Low"))

annotations.append(go.layout.Annotation(x=high_time,
                                        y=high_px,
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowcolor="Blue",
                                        arrowsize=2,
                                        arrowwidth=2,
                                        text="High"))

layout = dict(
        title="Stock Price Graph",
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title( text="Date")), 
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title( text="Price of stock")),         
        width=1000,
        height=800,
        annotations=annotations

)
data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])]
figSignal = go.Figure(data=data, layout=layout)
figSignal.show()




#STARING OUR SUPPORT VECTOR REGRESSION
#analysing the closing data for last 20 days before the current date
for i in range(1, 20, 1):
    spy.loc[:,'Close Minus' + str(i)] = spy['Close'].shift(i)
    spy20 = spy[[x for x in spy.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]
#reversing the model as we want consecutive data
spy20 = spy20.iloc[:,::-1]
#importing our support data machine
from sklearn.svm import SVR
clf = SVR(kernel='linear')
#Using the first 1000 days data out of the 4066 days data to perform regression
#Setting the training dataset
X_Training = spy20[:-1000]
Y_Training = spy20['Close'].shift(-1)[:-1000]
#Using the above training data to predict the data for 50 days
X_test = spy20[-50:]
Y_test = spy20['Close'].shift(-1)[-50:]
#Using the inbuilt function to perform regression
model = clf.fit(X_Training,Y_Training)
pred = model.predict(X_test)
#Converting the data to a column which corresponds the 50 days data using DataFrame
tf = pd.DataFrame(list(zip(pred)), columns=['Predicted Next Day Close'], index = Y_test.index)
#Merging the two Data Frames
xyz['Predicted Next Day Close'] = tf.iloc[0:50].values
print(xyz)






#Plotting the Trend of stock and prediction model vs real model
plt.rcParams["figure.figsize"] = (15,10)
plt.subplot(2, 1, 1)
plt.plot(spy_high, color='k')
plt.plot(spy_vol, color='r')
plt.title('Trends')
plt.subplot(2, 1, 2)
plt.plot(xyz['Predicted Next Day Close'],color='k')
plt.plot(xyz['Close'],color='r')
plt.show()





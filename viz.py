import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np

class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, Render_range):
        self.volume = deque(maxlen=Render_range)
        self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()


    # Render the environment to the screen
    def render(self, time, open, high, low, close, volume, net_worth, trades):

        self.volume.append(volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        time = mpl_dates.date2num([(pd.to_datetime(time,  unit='ms'))])[0]
        self.render_data.append([time, open, high, low, close])

        time_render_range = [i[0] for i in self.render_data]

        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/(24*60), colorup='green', colordown='red', alpha=0.8)



        self.ax2.clear()
        self.ax2.fill_between(time_render_range, self.volume, 0)

        self.ax3.clear()
        self.ax3.plot(time_render_range, self.net_worth, color="blue")

        self.ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d %H:%M'))
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:,1:])
        maximum = np.max(np.array(self.render_data)[:,1:])
        range = maximum - minimum

        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['time'], unit='ms')])[0]
            if trade_date in time_render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low']- range*0.02
                    ycoords = trade['Low'] - range*0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High'] + range*0.02
                    ycoords = trade['High'] + range*0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")


                try:
                    self.ax1.annotate('{0:.2f}'.format(trade['Reward']), (trade_date, high_low), xytext=(trade_date, ycoords),
                                               bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
                except:
                    pass

        self.ax2.set_xlabel('Time')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')
        self.fig.tight_layout()

#         # Show the graph without blocking the rest of the program
#         plt.show(block=False)
#         # Necessary to view frames before they are unrendered
#         plt.pause(0.001)

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape((1600, 3200, 3))

        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot",image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
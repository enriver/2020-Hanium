# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:21:45 2020

@author: seungjun
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from shutil import copyfile, move
from mpl_finance import candlestick2_ochl
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_datareader as web
from mpl_finance import candlestick_ohlc
from mpl_finance import volume_overlay





def ohlc2cs(df, seq_len, dimension, add_v):
    # python preprocess.py -m ohlc2cs -l 20 -i stockdatas/EWT_testing.csv -t testing
    print("Converting olhc to candlestick")

    df = df
    plt.style.use('dark_background')
    df.reset_index(inplace=True)

    figs = np.zeros((len(df)-1, dimension, dimension, 3))
    labels = []
    for i in range(0, len(df)-1):
        # ohlc+volume
        c = df.loc[i:i + int(seq_len) - 1, :]
        c_ = df.loc[i:i + int(seq_len), :]
        if len(c) == int(seq_len):
            my_dpi =100
            fig = plt.figure(figsize=(dimension / my_dpi, dimension / my_dpi), dpi=my_dpi)
            ax1 = fig.add_subplot(1, 1, 1)
            candlestick2_ochl(ax1, c['Open'], c['Close'], c['High'],c['Low'], width=1,colorup='#77d879', colordown='#db3f3f')

            ax1.grid(False)

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis('off')

            if add_v == True:
                ax2 = fig.add_subplot(2,1,1)
                volume_overlay(ax2, opens = c['Open'],closes= c['Close'], volumes = c['Volume'],colorup='#77d879', colordown='#db3f3f',width=1)

                ax2.grid(False)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax2.axis('off')
                

                
            # create the second axis for the volume bar-plot
            # Add a seconds axis for the volume overlay

            
                      
        starting = c_["Close"].iloc[-2]
        endvalue = c_["Close"].iloc[-1]
        if endvalue > starting :
            label = 1
        else :
            label = 0
        labels.append(label)

        fig.canvas.draw()
        fig.savefig('sample.png')
        fig_np = np.array(fig.canvas.renderer._renderer)
        figs[i] = fig_np[:,:,:3]
        plt.close(fig)
        # normal length - end

    print("Converting olhc to candlestik finished.")
    return figs, labels
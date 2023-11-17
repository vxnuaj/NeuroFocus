##This is the updated version of @Ahnaaf Khan's and Surya Sure's Neurofeedback Program | Updated to use the latest version of BrainFlow

##Header

import time
import sys

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.animation import FuncAnimation

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

##Main Function Setup

def main(i):
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value ##If you're using hardware, swap with the board you're using. Reference the BrainFlow Docs
    params.serial_port = 'COM5'
    board = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp = BoardShim.get_timestamp_channel(board_id)
    
    board.prepare_session()
    board.start_stream()
    style.use('fivethirtyeight')
    plt.title("Live EEG stream from Brainflow", fontsize=15)
    plt.ylabel("Data in millivolts", fontsize=15)
    plt.xlabel("\nTime", fontsize=10)
    keep_alive = True

    eeg1 = [] 
    eeg2 = []
    eeg3 = []
    eeg4 = []
    timex = [] 

##Data Acquisition and Filtering

    while keep_alive == True:

        while board.get_board_data_count() < 250: ##shouldn't it be 200 for the ganglion board?
            time.sleep(0.005)
        data = board.get_current_board_data(250)

        
        eegdf = pd.DataFrame(np.transpose(data[eeg_channels]))
        eegdf_col_names = ["ch1", "ch2", "ch3", "ch4"]
        eegdf.columns = [f"ch{i}" for i in range(1, len(eegdf.columns) + 1)] ##Placeholder for SYNTHETIC_BOARD; When you use real hardware, input eegdf.columns = eegdf_column_names
       
        timedf = pd.DataFrame(np.transpose(data[timestamp]))

        print("EEG Dataframe")
        print(eegdf)          

        for count, channel in enumerate(eeg_channels):
          
            if count == 0:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  
                DataFilter.perform_bandpass(data[channel], sampling_rate, 11.0, 31.0, 4,
                                            FilterTypes.BESSEL.value, 0)  
            if count == 1:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandpass(data[channel], sampling_rate, 11.0, 31.0, 4,
                                            FilterTypes.BESSEL.value, 0) 
            if count == 2:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  
                DataFilter.perform_bandpass(data[channel], sampling_rate, 11.0, 31.0, 4,
                                            FilterTypes.BESSEL.value, 0)  
            if count == 3:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  
                DataFilter.perform_bandpass(data[channel], sampling_rate, 11.0, 31.0, 4,
                                            FilterTypes.BESSEL.value, 0) 


##BrainFlow Machine Learning Model and the Focus / Restfulness Calculation
## In terms of the BrainFlow Library, Focus = Mindfulness

        bands = DataFilter.get_avg_band_powers(
            data, eeg_channels, sampling_rate, True)
        feature_vector = np.concatenate((bands[0], bands[1]))


        mindfulness_params = BrainFlowModelParams(
            BrainFlowMetrics.MINDFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        mindfulness = MLModel(mindfulness_params)
        mindfulness.prepare()
        print('Focus: %f' % mindfulness.predict(feature_vector))
        mindfulness_measure = mindfulness.predict(feature_vector)
        mindfulness.release()


        restfulness_params = BrainFlowModelParams(
            BrainFlowMetrics.RESTFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        restfulness = MLModel(restfulness_params)
        restfulness.prepare()
        print('Relaxation: %f' % restfulness.predict(feature_vector))
        restfulness_measure = restfulness.predict(feature_vector)
        restfulness.release()


        eeg1.extend(eegdf.iloc[:, 0].values) 
        eeg2.extend(eegdf.iloc[:, 1].values) 
        eeg3.extend(eegdf.iloc[:, 2].values) 
        eeg4.extend(eegdf.iloc[:, 3].values)
        timex.extend(timedf.iloc[:, 0].values) 

        plt.cla()
   
        plt.plot(timex, eeg1, label="Channel 1", color="red")
        plt.plot(timex, eeg2, label="Channel 2", color="blue")
        plt.plot(timex, eeg3, label="Channel 3", color="orange")
        plt.plot(timex, eeg4, label="Channel 4", color="purple")
        plt.tight_layout()
        keep_alive = False 

##Focus and Relaxation Feedback
      
        if mindfulness_measure >= 0.5:
            print("Supercharged!")
        else:
            print("Focus on what matters most.")
        
        ##if restfulness_measure >= 0.5:
            ##print("Relaxed")
        ##else:
           ## print("Let's relax...") 

    board.stop_stream()
    board.release_session()

##Matplotlib Animation

ani = FuncAnimation(plt.gcf(), main, interval=1000)
plt.tight_layout()
plt.autoscale(enable=True, axis="y", tight=True)
plt.show()
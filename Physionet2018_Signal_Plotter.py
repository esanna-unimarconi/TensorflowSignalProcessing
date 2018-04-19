"""
Physionet Challenge 2018 Signal Plotter

@description: execute this file to plot a signal

@author: Enrico Sanna - Unimarconi
@project: hhttps://github.com/esanna-unimarconi/TensorflowSignalProcessing/
@create-date:19/04/2018

"""
#librart to read Matlab v4 Files
#http://wfdb.readthedocs.io/en/latest/wfdb.html
import wfdb
#suppress hardware configuration warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#directory of the record
filename='M:/training/tr03-0005/tr03-0005'
#sample interval
sampFrom=0
sampTo=500

#read the record
#channels label ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
#Signal      Name	Units	        Signal Description
#SaO2	    %	                    Oxygen saturation
#ABD	        µV	                    Electromyography, a measurement of abdominal movement
#CHEST	    µV	                    Electromyography, measure of chest movement
#Chin1-Chin2	µV	                    Electromyography, a measure of chin movement
#AIRFLOW	    µV	                    A measure of respiratory airflow
#ECG	        mV	                    Electrocardiogram, a measure of cardiac activity
#E1-M2	    µV	                    Electrooculography, a measure of left eye activity
#O2-M1	    µV	                    Electroencephalography, a measure of posterior activity
#C4-M1	    µV	                    Electroencephalography, a measure of central activity
#C3-M2	    µV	                    Electroencephalography, a measure of central activity
#F3-M2	    µV	                    Electroencephalography, a measure of frontal activity
#F4-M1	    µV	                    Electroencephalography, a measure of frontal activity
#O1-M2	    µV	                    Electroencephalography, a measure of posterior activity

record = wfdb.rdrecord(filename, sampfrom=sampFrom, sampto=sampTo, channels=[0,1,2,3,4,5,6,7,8,9,10,11,12])

#plot signals
wfdb.plot_wfdb(record, plot_sym=True,time_units='seconds', title='Sample '+str(filename)+' time ( '+ str(sampFrom)+ ','+str(sampTo)+')',figsize=(10,9))

#plot entire record file
#wfdb.plot_all_records(directory='M:/training/tr03-0005/')
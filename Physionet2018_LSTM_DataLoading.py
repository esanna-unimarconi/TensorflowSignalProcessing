"""
Pysionet2018_LSTM_DataLoading

@description: A Class with function to read Physionet 2018 Challenge Dataset

@author: Enrico Sanna - Unimarconi
@project: hhttps://github.com/esanna-unimarconi/TensorflowSignalProcessing/
@create-date:18/04/2018
"""
import os
import numpy as np
# librart to read Matlab v4 Files
# http://wfdb.readthedocs.io/en/latest/wfdb.html
import wfdb
# library to easy manipulate Matlab v7.3 Files (HDF5), base on h5py.
# https://github.com/frejanordsiek/hdf5storage
import hdf5storage


class Pysionet2018_LSTM_DataLoading:

    def __init__(self, baseDirName="M:\\", currentDirName="none"):
        self.baseDirName = baseDirName
        self.currentDirName = currentDirName
        self.sample_from = 0
        if (currentDirName == "none"): self.next_record_directory()
        self.resetSampleFrom()

    def resetSampleFrom(self):
        self.sample_from = 0

    def getCurrentDirName(self):
        return self.currentDirName

    def getSampleFrom(self):
        return self.sample_from

    '''
        change attribute state with next record on the dataset
    '''

    def next_record_directory(self):
        training_directory = str(self.baseDirName + "training\\")
        trovata = 0
        for dirs in os.listdir(training_directory):
            if (not self.currentDirName.startswith("tr")):
                if (dirs.startswith("tr")):
                    self.currentDirName = dirs
                    # print("imposto directory " + dirs)
                    break
                # else: print("scarto directory "+dirs)
            else:
                if (trovata == 1):
                    self.currentDirName = dirs
                    self.resetSampleFrom()
                    # print("imposto directory " + dirs)
                    break
                else:
                    if (self.currentDirName == dirs and trovata == 0): trovata = 1
        print("Cambio record file: " + training_directory + "\\" + self.currentDirName)
        return self.currentDirName

    '''
        test function for print arousal vector
    '''

    def printArousalFile(self, filename, sample_from, signals_max_size=0, depth=10):
        arousalDataRecord = hdf5storage.loadmat(filename + '-arousal.mat')
        arousalArray = arousalDataRecord["data"][0][0][0][0]
        arousalDataRecord = hdf5storage.loadmat(filename + '-arousal.mat')
        arousalArray = arousalDataRecord["data"][0][0][0][0]
        # print("Arousal File " + str(filename) + " total size: " + str(arousalArray.size))
        signals_size = arousalArray.size
        # limit Array to requested size
        if (signals_max_size != 0): signals_size = min(signals_size, signals_max_size)
        # I discard firsts depth size
        arousalArray = arousalArray[sample_from + depth:sample_from + signals_size]
        # print("Arousal File " + str(filename) + " sample size: " + str(signals_size))
        arousalLabels = np.zeros((signals_size - depth, 3))
        i = 0
        for element in arousalArray:
            if element == 0: arousalLabels[i, 0] = 1; print("ciclo " + str(i) + " messo a zero");
            if element == 1: arousalLabels[i, 1] = 1
            if element == -1: arousalLabels[i, 2] = 1;print("ciclo " + str(i) + " messo a  - uno")
            i = i + 1

    '''
           extra signal from file and return signals,fields
           signals is a multi-array with signals
           fields is a dict with field names and unit of measurement
    '''

    def extractSignal(self, filename, sample_from, signals_max_size=0, depth=10):
        """
        To extract signals from training dataset
        @filename: filepath of the subject
        """
        # reading arousal datafile, goal of the challenge
        arousalDataRecord = hdf5storage.loadmat(filename + '-arousal.mat')
        arousalArray = arousalDataRecord["data"][0][0][0][0]
        # print("Arousal File " + str(filename) + " total size: " + str(arousalArray.size))
        signals_size = arousalArray.size
        # limit Array to requested size
        if (signals_max_size != 0): signals_size = min(signals_size, signals_max_size)
        # I discard firsts depth size
        arousalArray = arousalArray[sample_from + depth:sample_from + signals_size]
        # print("Arousal File " + str(filename) + " sample size: " + str(signals_size))

        arousalLabels = np.zeros((signals_size - depth, 3))
        i = 0
        for element in arousalArray:
            if element == 0: arousalLabels[i, 0] = 1;  # print("messo a zero")
            if element == 1: arousalLabels[i, 1] = 1
            if element == -1: arousalLabels[i, 2] = 1;  # print("messo a  - uno")
            i = i + 1
        # sampling a file from training dataset
        # ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
        # (channel 12 = ECG)
        signals, fields = wfdb.rdsamp(filename, sampfrom=sample_from, sampto=sample_from + signals_size,
                                      channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        # print("Signal Fields " + str(fields))
        return signals, arousalLabels, signals_size

    '''
        function to extract next batch-size dimension list o arrays
    '''

    def train_next_batch(self, batch_size, depth):
        filename = self.baseDirName + "training\\" + self.currentDirName + "\\" + self.currentDirName
        # self.printArousalFile( filename, self.sample_from, 4000000, 10)
        # exit(0)
        signals, arousalLabels, signals_size = self.extractSignal(filename, self.sample_from, batch_size, depth)
        # per ora campiono solo i primi 4 milioni di valori per record
        # print("Sample from: "+str(self.sample_from))
        if self.sample_from > 4000000:
            self.next_record_directory()
        else:
            self.sample_from = self.sample_from + batch_size

        # signals = np.zeros(depth,13)
        # arousalLabels = np.zeros(depth, 1)
        return signals, arousalLabels


'''
unit test of the class
'''
# loader = Pysionet2018_LSTM_DataLoading(currentDirName="tr03-0029")
# directory = loader.next_record_directory()
# print("Nuovo Record: "+str(directory))

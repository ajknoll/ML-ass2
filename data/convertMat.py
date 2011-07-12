#!/usr/bin/env python
# -*- coding: utf_8 -*-

"""
Converts an ALLEEG structure from a .mat file, as produced by EEGLAB/Matlab,
to a more useful data structure.
"""

import argparse
import sys
import cPickle
import scipy.io as sio
import numpy

# constants
PICKLE_BINARY = 2

# Call ExperimentData(filename) to turn the .mat files into instances of this
# class
class ExperimentData:
   CONDITION_COUNT = 4

   def __init__(self, alleegFilename = None):
      if (alleegFilename != None):
         matFile = sio.loadmat(alleegFilename, struct_as_record = True)
         alleeg = matFile['ALLEEG']

         # preallocate appropriately sized list
         subjectTotal = int(numpy.ceil(len(alleeg[0]) / self.CONDITION_COUNT))
         self.matrix = [[None for condition in range(self.CONDITION_COUNT)] 
                        for subject in range(subjectTotal)]
         # fill list
         subjectCount = 0
         minEpochSize = sys.maxint # for later epoch size checking
         currentSubject = int(alleeg[0][0]['subject'][0])
         for record in alleeg[0]:
            # convert eeglab format to python class
            newRecord = TaskRecording(record)
            # insert in matrix
            if newRecord.subject != currentSubject:
               currentSubject = newRecord.subject
               subjectCount += 1
            self.matrix[subjectCount][newRecord.condition] = newRecord
            minEpochSize = min(minEpochSize, newRecord.epochSize)

         # split any oversized epochs into minimum-epoch-size segments
         for subject in self.matrix:
            for task in subject:
               task.splitEpochs(minEpochSize)


   # Not sure how these work yet...
   def loadData(self, filename):
      self.matrix = cPickle.load(open(filename, 'rb'))

   def saveData(self, filename):
      cPickle.dump(self.matrix, open(filename, 'wb'), PICKLE_BINARY)


# TaskRecording contains the details and data associated with a single recording
# i.e. one subject x task level unit.
class TaskRecording:
   def __init__(self, record):
      # subject id 
      self.subject      = int(record['subject'][0])
      # task condition
      self.condition    = self._taskNameToInt(record['condition'][0])
      # number of recording channels
      self.nChans       = record['nbchan'][0][0]
      # channel labels
      self.chanLabels   = [channel['labels'][0] for channel in record['chanlocs'][0]]
      # length of an individual epoch in seconds
      self.epochSize    = record['xmax'][0][0] - record['xmin'][0][0]
      # sampling rate of recording in Hertz (samples/second)
      self.sampleRate   = record['srate'][0][0]
      # time series, channel x sample [x epoch, if more than 1] matrix
      self.data         = record['data']
      # list of recorded "events", such as slide starts and mouse clicks during
      # the experiment. Latency is relative to the enire record, not the
      # individual epoch.
      self.events       = [ {'latency': event['latency'][0][0],
                             'name':    self._eventName(event['type'][0]),
                             'epoch':   event['epoch'][0][0]
                            } for event in record['event'][0] ]
      if len(self.data.shape) < 3:
         self.nEpochs = 1
      else:
         self.nEpochs = self.data.shape[2]

   #def epochEvents(self, epoch):
    #  if (epoch >= len(self.epoch

   def splitEpochs(self, epochSize):
      assert(epochSize > 0)
      
      if (epochSize < self.epochSize):
         newEpochSampleCount = int(epochSize * self.sampleRate)
         newEpochCount = self.data[0].size / newEpochSampleCount
         if newEpochSampleCount * newEpochCount != self.data[0].size:
            if (self.data.ndim > 2):
               self.data = self.data[:, :-(self.data.shape[1] % newEpochSampleCount), :]
            else:
               self.data = self.data[:, :-(self.data.shape[1] % newEpochSampleCount)]
         self.data = self.data.reshape((self.data.shape[0], newEpochSampleCount, newEpochCount))
         self.epochSize = epochSize
         self.nEpochs = newEpochCount
         # TODO: implement event['epoch'] updating

   def _eventName(self, event): # internal event number -> name mapping
      nameDic = {'0': 'experimentStart',
                 '1': 'experimentEnd',
                 '3': 'slideStart',
                 '4': 'slideEnd',
                 '100': 'leftClick',
                 '101': 'middleClick',
                 '102': 'rightClick',
                 '200': 'emotivBlink',
                 '255': 'emotivError'
                }

      return(nameDic[event])

   def _taskNameToInt(self, taskName): # internal task name -> index mapping
      taskDic = {'open': 0,
                 '1':    1,
                 '2':    2,
                 '3':    3 
                }
      return(taskDic[taskName])


def main(*args):
   """
   This doesn't work, yet. Just use ExperimentData.__init__ from another module
   or interactive session.
   """
   parser = argparse.ArgumentParser(
         description = 'Convert eeglab ALLEEG structure from .mat to pickled python.')
   parser.add_argument('inputFile')
   parser.add_argument('outputFile')
   argsParsed = parser.parse_args(args[1:])
   
   import convertMat # required for pickle to correctly isolate the class from main
   data = convertMat.ExperimentData(argsParsed.inputFile)
   data.saveData(argsParsed.outputFile)

if __name__ == "__main__":
   main(*sys.argv)

#!/usr/bin/env python
# -*- coding: utf_8 -*-

""" 
Process signals to extract features for classification/learning.
"""

from data.convertMat import ExperimentData, TaskRecording
import numpy

class SignalLearn:

   def getRmsList(self, data):
      assert(data.__class__ == ExperimentData)
      for subject in data.matrix:
         for task in subject:
            if task.nEpochs == 1:
               rmsList = [rootMeanSquare(channel) for channel in task.data]
               sample.append(rmsList)
               classes.append(task.condition)
            else:
               for epoch in seq(task.nEpochs):
                  rmsList = [rootMeanSquare(channel) for channel in task.data[:, :, epoch]]
                  sample.append(rmsList)
                  classes.append(task.condition)


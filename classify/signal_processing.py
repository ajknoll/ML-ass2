#!/usr/bin/env python
# -*- coding: utf_8 -*-

""" 
Process signals to extract features for classification/learning.
"""

from data.convertMat import ExperimentData, TaskRecording
import numpy

class SignalLearn:

   def getRmsList(self, data):
      """
      Given an ExperimentData object data,
      produces a list of root mean square (RMS) values per epoch. The feature
      dimension is a list of signal channels. There is a corresponding list of
      class labels for the RMS values.

      Returns:
      sample   - a list of [epoch, channel] RMS values
      classes  - corresponding list of class labels
      """
      assert(data.__class__ == ExperimentData)
      sample = []
      classes = []
      for subject in data.matrix:
         for task in subject:
            if task.nEpochs == 1:
               rmsList = [self.rootMeanSquare(channel) for channel in task.data]
               sample.append(rmsList)
               classes.append(task.condition)
            else:
               for epoch in seq(task.nEpochs):
                  rmsList = [self.rootMeanSquare(channel) for channel in task.data[:, :, epoch]]
                  sample.append(rmsList)
                  classes.append(task.condition)
      return sample, classes

   def rootMeanSquare(self, arrayLike):
      """
      Returns the root mean square of the elements in arrayLike.
      """
      array = numpy.asanyarray(arrayLike)
      rms = numpy.sqrt(numpy.sum([numpy.square(elem) for elem in array]))
      return rms

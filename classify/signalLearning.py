#!/usr/bin/env python
# -*- coding: utf_8 -*-

""" 
Process signals to extract features for classification/learning.
Then, apply learning algorithms in a way that gives informative output!
"""

from data.convertMat import ExperimentData, TaskRecording
import scikits.learn.cross_val as cross_val
import numpy

class SignalLearn:

   def rootMeanSquare(self, arrayLike):
      """
      Returns the root mean square of the elements in arrayLike.
      """
      array = numpy.asanyarray(arrayLike)
      rms = numpy.sqrt(numpy.sum(numpy.square(array)))
      return rms

   def getRmsList(self, data, subjectId = None):
      """
      Given an ExperimentData object data,
      produces a list of root mean square (RMS) values per epoch. The feature
      dimension is a list of signal channels. There is a corresponding list of
      class labels for the RMS values.

      Returns:
      sample   - a list of [epoch, channel] RMS values
      subjects - corresponding list of subject ids
      classes  - corresponding list of class labels
      """
      assert(data.__class__ == ExperimentData)
      sample = []
      classes = []
      for subject in data.matrix:
         for task in subject:
            if subjectId == None or task.subject == subjectId:
               if task.nEpochs == 1:
                  rmsList = [self.rootMeanSquare(channel) for channel in task.data]
                  sample.append(rmsList)
                  classes.append(task.condition)
               else:
                  for epoch in range(task.nEpochs):
                     rmsList = [self.rootMeanSquare(channel) for channel in task.data[:, :, epoch]]
                     sample.append(rmsList)
                     classes.append(task.condition)
      return sample, classes

   def leaveOneOut(self, sample, classes, learner, classifier):
      """
      Performs leave-one-out cross-validation on the given learner/classifier
      pair, given a [examples, features] sample array and the associated
      [examples] classes array.

      Learner and classifier are assumed to be functions that can be called as:
      learner(sample, classes)
      result = classifier(sample) 
      """
      assert(len(sample) == len(classes))
      sampleSize = len(sample)
      ndSample = numpy.asanyarray(sample)
      ndClasses = numpy.asanyarray(classes)

      crossMatrix = cross_val.LeaveOneOut(sampleSize)
      resultsVector = numpy.zeros([sampleSize])
      for trainIndex, testIndex in crossMatrix:

         # Assign train/test sets to arrays
         trainSample = ndSample[trainIndex]
         trainClasses = ndClasses[trainIndex]
         testSample = ndSample[testIndex]
         # testClasses = ndClasses[testIndex]

         # Learn the training set
         learner(trainSample, trainClasses)
         # Attempt classification and store results
         resultsVector[testIndex] = classifier(testSample)

      return resultsVector


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
      sample  - a list of [epoch, channel] RMS values
      classes - corresponding list of class labels
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

   def getSpectralDecomp(self, data, minFreq, maxFreq, freqStep, subjectId = None):
      """
      Given an ExperimentData object data,
      produces a spectral decomposition of each contained time-series. The 
      feature dimension is channels x frequency buckets.

      Returns:
      sample  - a list of [epoch, channel, frequency bucket] power densities
      classes - a corresponding list of class labels

      """

   def _crossValAccuracy(self, crossValResults, trueClasses):
      ndResults = numpy.asanyarray(crossValResults)
      ndClasses = numpy.asanyarray(trueClasses)

      correct = 0
      for i in range(len(ndResults)):
         if (ndResults[i] == ndClasses[i]):
            correct += 1

      if correct > 0:
         return float(correct) / len(ndResults)
      else:
         return 0

   def kFoldVal(self, sample, classes, learner, classifier, k = 10):
      """
      Peforms k-fold validation on the given learner/classifier pair, given
      an [examples, features] sample array and the associated [examples]
      classes array.

      Learner and classifier are assumed to be functions that can be called as:
      learner(sample, classes)
      result = classifier(sample) 
      """

      assert(len(sample) == len(classes))
      sampleSize = len(sample)
      ndSample = numpy.asanyarray(sample)
      ndClasses = numpy.asanyarray(classes)

      crossMatrix = cross_val.KFold(sampleSize, k)
      resultsVector = numpy.zeros([sampleSize])
      progress = 0
      progressGranularity = 1
      print "Starting cross-validation..."
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

         progress += 1
         #print progress, progressGranularity
         if progress % progressGranularity == 0:
            print progress, "/", k, " done"

      accuracy = self._crossValAccuracy(resultsVector, ndClasses)
      print "Cross-validation accuracy: ", accuracy

      return resultsVector, accuracy



   def leaveOneOut(self, sample, classes, learner, classifier):
      """
      Performs leave-one-out cross-validation on the given learner/classifier
      pair, given a [examples, features] sample array and the associated
      [examples] classes array.

      Learner and classifier are assumed to be functions that can be called as:
      learner(sample, classes)
      result = classifier(sample) 
      """
      PROGRESS_FACTOR = 200

      assert(len(sample) == len(classes))
      sampleSize = len(sample)
      ndSample = numpy.asanyarray(sample)
      ndClasses = numpy.asanyarray(classes)

      crossMatrix = cross_val.LeaveOneOut(sampleSize)
      resultsVector = numpy.zeros([sampleSize])
      progress = 0
      progressGranularity = 1
      if sampleSize >= PROGRESS_FACTOR:
         progressGranularity = sampleSize / PROGRESS_FACTOR
      print "Starting cross-validation..."
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

         progress += 1
         #print progress, progressGranularity
         if progress % progressGranularity == 0:
            print progress, "/", sampleSize, " done"

      accuracy = self._crossValAccuracy(resultsVector, ndClasses)
      print "Cross-validation accuracy: ", accuracy

      return resultsVector, accuracy

#!/usr/bin/env python
# -*- coding: utf_8 -*-

import data.convertMat as convert
import classify.signalLearning as sigLearn
import classify.spectrum as spectrum
import numpy
import scikits.learn.svm as svm
import scikits.learn.cross_val as cross_val

"""
Some examples of the scripts used in IPython sessions.
runSpectralCrossValidation, if passed for example:
   bins = 32
   learner = svm.LinearSVC()
   crossValidator = sigLearn.stratifiedKFoldVal
   k = 10
will produce results similar to those found in our report.

Of course, data needs to be initialised first- e.g.:
   data = convert.ExperimentData("data/subject_x_task.mat")
   for subject in data.matrix:
      for task in subject:
         task.splitEpochs(1)
Which will produce a dataset with a large number of 1 second 
non-overlapping epochs/segments.
"""

def checkClassAccuracies(accuracies):
    count = 0
    for accuracy in accuracies.flatten():
       if accuracy > 0.5:
          count += 1
    return numpy.true_divide(count, accuracies.size)

def runSpectralCrossValidation(data, bins, learner, crossValidator, k = None)
   sigLearnInstance = sigLearn.SignalLearn()
   perSubjectAccuracy = numpy.zeros(len(data.matrix))
   classAccuracies = numpy.zeros((len(data.matrix), 4))
   for i, subject in zip(range(len(data.matrix)), data.matrix):
      print "Starting subject {0}/{1}".format(i + 1, len(data.matrix))
      sample, classes = sigLearnInstance.getSpectralDecomp(data, bins, i)
      if k == None:
         results, perSubjectAccuracy[i], classAccuracies[i] = \
               crossValidator(sample, classes, learner.fit, learner.predict)
      else:
         results, perSubjectAccuracy[i], classAccuracies[i] = \
               crossValidator(sample, classes, learner.fit, learner.predict, k)

   return perSubjectAccuracy, classAccuracies

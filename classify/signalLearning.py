#!/usr/bin/env python
# -*- coding: utf_8 -*-

""" 
Process signals to extract features for classification/learning.
Then, apply learning algorithms in a way that gives informative output!
"""

from data.convertMat import ExperimentData, TaskRecording
import spectrum
import scikits.learn.cross_val as cross_val
import numpy
import time

class SignalLearn:

   def rootMeanSquare(self, arrayLike):
      """
      Returns the root mean square of the elements in arrayLike.
      """
      array = numpy.asanyarray(arrayLike)
      rms = numpy.sqrt(numpy.sum(numpy.square(array)))
      return rms

   def getRmsList(self, data, subjectIndex = None):
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
      for subject, index in zip(data.matrix, range(len(data.matrix))):
         for task in subject:
            if subjectIndex == None or index == subjectIndex:
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
   
   def spectrumFilter(self, spec, freqs, lowCutoff, highCutoff):
      if len(spec) != len(freqs):
         raise ValueError("spec and freqs must be of equal length")

      filterMatrix = numpy.empty_like(freqs, dtype = 'bool')
      if lowCutoff != None or highCutoff != None:
         for i, s, f in zip(range(len(filterMatrix)), spec, freqs):
            if (lowCutoff != None and f < lowCutoff) or (highCutoff != None and f > highCutoff):
               filterMatrix[i] = False
            else:
               filterMatrix[i] = True
      return spec[filterMatrix], freqs[filterMatrix]

   def getSpectralDecomp(self, data, numBins, subjectIndex = None, lowCutoff = None, highCutoff = None):
      """
      Given an ExperimentData object data,
      produces a spectral decomposition of each contained time-series. The 
      feature dimension is channels x frequency buckets.

      Returns:
      sample  - a list of [epoch, channel, frequency buckets] power densities
      classes - a corresponding list of class labels
      """
      assert(data.__class__ == ExperimentData)
      assert(numBins != 0)
      sample = []
      classes = []
      for subject, index in zip(data.matrix, range(len(data.matrix))):
         for task in subject:
            if subjectIndex == None or index == subjectIndex:
               if task.nEpochs == 1:
                  spectra = []
                  for channel in task.data:
                     channelSpec, freqs = spectrum.solveSpectrum(channel, task.sampleRate)
                     #channelSpec, freqs = self.spectrumFilter(channelSpec, freqs, lowCutoff, highCutoff)
                     # The highest frequencies are most likely to be noise here,
                     # so we can trim those to fit evenly into bins.
                     if channelSpec.size % numBins != 0:
                        channelSpec = channelSpec[:-(channelSpec.size % numBins)]
                     channelSpec = spectrum.bin(channelSpec, numBins, method = 'sum')
                     spectra.append(channelSpec)
                  sample.append(spectra)
                  classes.append(task.condition)
               else:
                  for epoch in range(task.nEpochs):
                     spectra = []
                     for channel in task.data[:, :, epoch]:
                        channelSpec, freqs = spectrum.solveSpectrum(channel, task.sampleRate)
                        #channelSpec, freqs = self.spectrumFilter(channelSpec, freqs, lowCutoff, highCutoff)
                        # The highest frequencies are most likely to be noise here,
                        # so we can trim those to fit evenly into bins.
                        if channelSpec.size % numBins != 0:
                           channelSpec = channelSpec[:-(channelSpec.size % numBins)]
                        channelSpec = spectrum.bin(channelSpec, numBins, method = 'sum')
                        spectra.append(channelSpec)
                     sample.append(spectra)
                     classes.append(task.condition)
      return sample, classes

   def crossValAccuracy(self, crossValResults, trueClasses):
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

   def _perClassAccuracy(self, crossValResults, trueClasses):
      ndResults = numpy.asanyarray(crossValResults)
      ndClasses = numpy.asanyarray(trueClasses)

      classCorrect = numpy.zeros_like(numpy.unique(ndClasses))
      classTotal = numpy.zeros_like(classCorrect)
      classIndex = dict(zip(numpy.unique(ndClasses).tolist(), range(len(classTotal))))
      
      for result, cls in zip(ndResults, ndClasses):
         if result == cls:
            classCorrect[classIndex[cls]] += 1
         classTotal[classIndex[cls]] += 1

      accuracies = numpy.true_divide(classCorrect, classTotal)
      return accuracies
   
   def _flatten2D(self, arrayLike):
      if (arrayLike.ndim <= 2):
         return arrayLike
      else:
         return arrayLike.reshape((arrayLike.shape[0], arrayLike[0].size))

   def _crossVal(self, sample, classes, learner, classifier, crossValMatrix, progressGranularity):
      """
      Runs the actual cross-validation process, as well as printing progress to
      stdout.
      """
      print "Starting cross-validation..."
      progress = 0
      resultsVector = numpy.zeros(len(sample))
      classAccuracies = numpy.zeros((len(crossValMatrix), len(numpy.unique(classes))))
      for trainIndex, testIndex in crossValMatrix:
         # Assign train/test sets to arrays
         trainSample = self._flatten2D(sample[trainIndex])
         trainClasses = classes[trainIndex]
         testSample = self._flatten2D(sample[testIndex])
         testClasses = classes[testIndex]

         # Learn the training set
         learner(trainSample, trainClasses)
         # Attempt classification and store results
         results = classifier(testSample)
         resultsVector[testIndex] = results
         classAccuracies[progress] = self._perClassAccuracy(results, testClasses)

         progress += 1
         if progress % progressGranularity == 0:
            print "{0}: {1}/{2} done".format(time.strftime("%H:%M:%S"), progress, len(crossValMatrix)) 
      return resultsVector, classAccuracies
   
   def _printAccuracy(self, meanAccuracy, classAccuracy):
      print "Cross-validation accuracy:", meanAccuracy
      print "Per-class cross-validation accuracy:", classAccuracy

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

      crossValMatrix = cross_val.KFold(sampleSize, k)
      resultsVector, classResults = self._crossVal(ndSample, ndClasses, learner, classifier, crossValMatrix, 1)
      accuracy = self.crossValAccuracy(resultsVector, ndClasses)
      classAccuracy = [numpy.rint(cls).sum() / cls.size for cls in numpy.transpose(classResults)]
      self._printAccuracy(accuracy, classAccuracy)

      return resultsVector, accuracy, classAccuracy

   def stratifiedKFoldVal(self, sample, classes, learner, classifier, k = 5):
      """
      Peforms stratified k-fold validation on the given learner/classifier pair, 
      given an [examples, features] sample array and the associated [examples]
      classes array.

      Learner and classifier are assumed to be functions that can be called as:
      learner(sample, classes)
      result = classifier(sample) 
      """
      assert(len(sample) == len(classes))
      sampleSize = len(sample)
      ndSample = numpy.asanyarray(sample)
      ndClasses = numpy.asanyarray(classes)

      crossValMatrix = cross_val.StratifiedKFold(ndClasses, k)
      resultsVector, classResults = self._crossVal(ndSample, ndClasses, learner, classifier, crossValMatrix, 1)
      accuracy = self.crossValAccuracy(resultsVector, ndClasses)
      classAccuracy = [numpy.rint(cls).sum() / cls.size for cls in numpy.transpose(classResults)]
      self._printAccuracy(accuracy, classAccuracy)

      return resultsVector, accuracy, classAccuracy
      

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

      crossValMatrix = cross_val.LeaveOneOut(sampleSize)
      if sampleSize >= PROGRESS_FACTOR:
         progressGranularity = sampleSize / PROGRESS_FACTOR
      else:
         progressGranularity = 1
      resultsVector, classResults = self._crossVal(ndSample, ndClasses, learner, classifier, crossValMatrix, progressGranularity)
      accuracy = self.crossValAccuracy(resultsVector, ndClasses)
      classAccuracy = [numpy.rint(cls).sum() / cls.size for cls in numpy.transpose(classResults)]
      self._printAccuracy(accuracy, classAccuracy)

      return resultsVector, accuracy, classAccuracy

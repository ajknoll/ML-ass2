#!/usr/bin/env python

import numpy as np
import scipy.fftpack as fftp

"""
A number of utility functions for extracting spectrums from
time series data.
"""


"""
calculates the spectral decomposition of a sample
Returns:
    spectrum
    bin labels
"""
def solveSpectrum(signal, sampleRate):

   if signal.ndim != 1:
      raise ValueError("Signal must be 1 dimensional")
   return (np.abs(fftp.rfft(signal)), fftp.rfftfreq(signal.size, 1./sampleRate))


"""
Reduce a signal into evenly sized bins by calculating sums or means
of groups of adjacent elements.
"""
def bin(signal, numBins, method='sum'):
   
   if signal.size % numBins != 0:
      raise ValueError("Signal not evenly divisible into requested number of bins")

   if signal.ndim != 1:
      raise ValueError("Signal must be 1 dimensional")

   if method == 'sum':
      return signal.reshape(-1, signal.size / numBins).sum(axis=1)
   elif method == 'mean':
      return signal.reshape(-1, signal.size / numBins).mean(axis=1)
   else:
      raise ValueError("Method must be one of 'sum' or 'mean'")

"""
Calculates Root mean squared of a given signal
"""
def RMS(signal):
    return np.sqrt(np.square(signal).sum())

"""
Normalises signal by centring it around 0 and correcting for expected deviation.
"""
def normalise(signal):
    mean = signal.mean(axis=0)
    Xc = signal-mean
    std = Xc.std(axis=0)
    return Xc/std

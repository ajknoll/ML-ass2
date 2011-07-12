#!/usr/bin/env python

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 2:
    print "Usage: graph.py <datafile>\n"
    sys.exit(0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data = np.loadtxt(sys.argv[1])
assert(data.shape == (3,54))
ax.scatter(data[0,0:17], data[1,0:17], data[2,0:17], c='r')
ax.scatter(data[0,18:35], data[1,18:35], data[2,18:35], c='g')
ax.scatter(data[0,35:], data[1,35:], data[2,35:], c='b')

plt.show()

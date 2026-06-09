# Murty's Algorithm Implementation Registry

This file is there to track the source, architecture, and baseline characteristics of each implementation of Murty's.

##lauziere
Source: https://github.com/lauziere/Murty
Language Used: Python 3
I think this one uses the idea of copying the original matrix and then creating the next subproblems by slapping 10^6 (infinity) to block paths. Then it just runs the SciPy solver completely from scratch. This one was the one that me and Professor had worked on in the past in improving our own version of the Murtys.


##FastMurty
Source: https://github.com/motrom/fastmurty
Language Used: C & Python for testing.
This one is the one that we have been implementing and looking at for a while. It is the one with a global matrix in memory. 


##MHT
Source: https://github.com/jonatanolofsson/mht
Language Used: C++ & Python
This one is part of a larger Multiple Hypothesis Tracking framework. From what I see it is implemented with C++ and then wrapped inside a Python interface.

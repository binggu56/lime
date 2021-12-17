# Author: Coline Devin
# File created: April 1, 2014
#
# This file uses the normal coordinates of the molecule to determine the change
# in normal coordinates between states.

import numpy as np

def calcDQ(gsEqCoords, gsFreqCoords, exEqCoords, exFreqCoords):
    """ Input is two lists of the form [eqCoords, (f1, f1coords), ... (fm, fm coords)]
        where the "coords" are lists of the form [x1, y1, z1, x2, y2,..., xn, yn, zn]
        for a molecule with n atoms and m vibrational modes. The ground state list
        comes first.
        Output is a list of tuples. Each tuple gives the ground state frequency,
        the excited state frequency, and the dQ for one mode
    """

    # Make a list of lists [[x1_1, ...zn_1], ..., [x1_m, ..., zn_m]]
    gsListOfQCoords = [coords for (f, coords) in gsFreqCoords]
    exListOfQCoords = [coords for (f, coords) in exFreqCoords]

    # Convert lists into vectors
    gsEqCartCoords = np.array(gsEqCoords)
    exEqCartCoords = np.array(exEqCoords)

    #print "gsEq", gsEqCartCoords
   # print "exEq", exEqCartCoords

    gsQCoords = np.array(gsListOfQCoords)
    exQCoords = np.array(exListOfQCoords)

    # Change in equilibirum cartesian coordinates 
    changeCartCoords = np.subtract(gsEqCartCoords, exEqCartCoords)
    #print "exQCoords", exQCoords
    #print "change", changeCartCoords

    # Convert Change to Q-coordinates
    #dQArray = np.dot(exQCoords, changeCartCoords)
    dQArray = np.dot(gsQCoords, changeCartCoords)
    dQList = list(dQArray)

    # Get frequencies
    gsFreqs = [f for (f,coords) in gsFreqCoords]
    exFreqs = [f for (f,coords) in exFreqCoords]

    # Return a list of tuples. Each tuple gives the ground state frequency,
    # the excited state frequency, and the dQ for one mode.
    return zip(gsFreqs,exFreqs,  dQList)

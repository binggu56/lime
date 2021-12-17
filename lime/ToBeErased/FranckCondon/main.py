import ParseFile as pf
import Dushinsky as d
import DiffFreqs as df
import Mode as m
import RecursiveModes as r
import ParallelModes as p

import sys
import getopt
import optparse

if __name__ == "__main__":
    gsfile = ''
    exfile = ''
    E_electronic = 0
    threshold = 0
    graph = "none"
    myopts, args = getopt.getopt(sys.argv[1:], "i:f:e:t:F:g:o:")
    output = None 

    # o == option
    # a == argument passed to the o
    READ_FILE = 1
    for o, a in myopts:
        if o == '-i':
            gsfile=a
        if o == '-f':
            exfile=a
        elif o == '-e':
            E_electronic = float(a)
        elif o == '-t':
            threshold = float(a)
        elif o == '-F':
            READ_FILE = a
            print "Setting READ_FILE to", a
        elif o == '-g':
            graph = a
        elif o == '-o':
            output = a
        else:
            print(("Usage: %s -i initialStateFile -f finalStateFile" +
                   " -e E_Electronic -t threshold [-g [curve or stick]") % sys.argv[0])
 
    # Display input and output file name passed as the args
    print (("Initial state file : %s, Final state file : %s,"+
            " electronic energy : %f,"+
            "threshold : %f") % (gsfile, exfile, E_electronic, threshold) )
    

    if READ_FILE == 1:
        print "Read File is", READ_FILE
        (gsEq, gsFreqCoords) = pf.parseNormalCoordinates(gsfile)
        (exEq, exFreqCoords) = pf.parseNormalCoordinates(exfile)
        
        freqsAndDQs = d.calcDQ(gsEq, gsFreqCoords, exEq, exFreqCoords)
        print freqsAndDQs
        # CHANGE BAKC TO EXFREQ
        modes = [m.Mode(gsFreq, gsFreq, dQ) for (gsFreq, exFreq, dQ) in 
                 freqsAndDQs]
        #modes = [modes[2]]


    else:
##### Change the variables below to run the code with a different set of modes. A mode is created by typing m.Mode(wi, wf, dQ) #####
        #  modes = [m.Mode(507.64, 507.64,0.374722838/0.5291711), m.Mode(897.05, 897.05, -0.203348073/0.5291711)]
        modes = [m.Mode(355.53798, 371.2870855649, 2.92714)]
        # m.Mode(770.120,770.120, 0.237238), m.Mode(793.482, 793.481, 0.0)]
        #modes = [m.Mode(355.5379891821,355.5379891821, 1.70338), m.Mode(770.1203772628,770.1203772628,1.83856 ), m.Mode(793.4813975867, 793.4813975867,0.0) ]
        #        modes = [m.Mode, 2.03397 )]
        E_electronic = 0
        #        modes = [modes[2]]

    print("Found %d modes" % (len(modes)))

    (energies, intensities, numpoints)  = r.genMultiModePoints(
        threshold, modes, E_electronic, 11)
    
    print "E len", len(energies), "I len", len(intensities)
    
    wide = [100]*numpoints
    med = [10]*numpoints
    skinny = [1]*numpoints
    
    energies.reverse()
    intensities.reverse()
    print("Found %d intensities" % len(intensities))
    for i in intensities:
        print i

    
    # sorted = sorted(zip(intensities, energies), reverse=True)\
#     sortedE = sorted(zip(energies, intensities))

#     print "Sorted by Energy"
#     for x in range(min(len(sortedE), 60)):
                      
#         print "E: ", sortedE[x][0], " I: ", sortedE[x][1]

#     sortedI = sorted(zip(intensities, energies), reverse=True)\

#     print
#     print "Sorted by Intensity"
#     for x in range(min(len(sortedI), 60)):          
#         print "E: ", sortedI[x][1], " I: ", sortedI[x][0]

    if graph != "none":
        import Plots as p

    if graph == "stick":
           p.plotSticks(energies, intensities, gsfile + " -> " + exfile)
    elif graph == "curve":
        p.plotSpectrum(energies, intensities, wide, gsfile + " -> " + exfile)
        print "Showing graph"
    raw_input("Press ENTER to exit ")

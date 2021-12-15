import Mode

# def parseFile(filename):
#     """ reads a file in the form of sampleInput.txt
#         returns a tuple (threshold, E_electronic, numModes, listOfModes)
#     """
#     f = open(filename)
#     input =  [l.split() for l in f]
#     thresh = float(input[0][0])
#     E_el = float(input[1][0])
#     numModes = int(input[3][0]) # Skipped a blank line in the file
#     currLine = 5
#     modes = []
#     for n in range(numModes):
#         if (currLine +2) < len(input):
#             print "currLine = ", currLine
#             gsFreq = float(input[currLine][0])
#             exFreq = float(input[currLine+1][0])
#             dQ = float(input[currLine+2][0])
#             currLine += 4 # Blank line between modes
#             mode = Mode.Mode(gsFreq, exFreq, dQ)
#             modes += [mode]
#         else:
#             raise EOFError("Incomplete mode on line " + str(currLine))
#     return (thresh, E_el, numModes, modes)

def parseNormalCoordinates(filename):
    """ read a file in the form InputFiles/ccl2_gs_mbpt_freq.normco.new
    """

    # Minimum frequency to be considered a vibrational mode
    freqThreshold = 20
    
    f = open(filename)
    splitLines = [l.split() for l in f]
    currLine = 1 # First line is a comment
    totLines = len(splitLines)
    nAtoms = 0  # number of atoms
    freqCoords = []
    eqCoords = []
    # Figure out number of atoms from "mass weighted coordinates" section
    while splitLines[currLine][0] != "%": # Keep going until "% frequency"
        eqCoords += splitLines[currLine]
        nAtoms += 1
        currLine += 1
    while currLine < totLines:
        #print "currline",  currLine
        if splitLines[currLine] == ["%", "frequency"]:
            currLine += 1
            try:
                freq = float(splitLines[currLine][0])
            except ValueError:
                freq = 0
            currLine += 2 # Skip comment
            if freq > freqThreshold:
                listOfCoords = []
                for a in range(nAtoms):
                    #print "atom", a
                    listOfCoords += [float(x) for x in splitLines[currLine]]
                    currLine += 1
                freqCoords += [(freq, listOfCoords)]
            else:
                currLine += nAtoms # If freq is too low, skip mode
    eqCoords = map(float, eqCoords)
    return (eqCoords, freqCoords)
(eq, fcoords) =  parseNormalCoordinates("InputFiles/ccl2_gs_mbpt_freq.normco.new")

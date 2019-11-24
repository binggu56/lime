from matplotlib import rc

def set_style(fontsize=12):
    import matplotlib.pyplot as plt
    import matplotlib 


    size = fontsize
    
    #fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    #'weight' : 'normal', 'size' : fontsize}
    
    fontProperties = {'family':'sans-serif','sans-serif':['Arial'],
    'weight' : 'normal', 'size' : fontsize}
        
    rc('font', **fontProperties)

    rc('text', usetex=False)

    plt.rc('xtick', color='k', labelsize='medium', direction='in')
    plt.rc('ytick', color='k', labelsize='medium', direction='in')
    plt.rc('xtick.major', size=4, pad=4)
    plt.rc('xtick.minor', size=2, pad=4)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


#    plt.rcParams['text.latex.preamble'] = [
###    r'\usepackage{time}',
#    r'\usepackage{tgheros}',    # helvetica font
#    r'\usepackage[]{amsmath}',   # math-font matching  helvetica
#    r'\usepackage{bm}',
##    r'\sansmath'                # actually tell tex to use it!
#    r'\usepackage{siunitx}',    # micro symbols
#    r'\sisetup{detect-all}',    # force siunitx to use the fonts
#    ]

    #rc('text.latex', preamble=r'\usepackage{cmbright}')



    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)

    #plt.rc('font', size=SIZE, weight='bold')  # controls default text sizes
    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x any y labels
    plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size)  # legend fontsize
    plt.rc('figure', titlesize=size)  # # size of the figure title
    plt.rc('axes', linewidth=1)

    #plt.rcParams['axes.labelweight'] = 'normal'

    plt.locator_params(axis='y', nticks=6)

    # the axes attributes need to be set before the call to subplot
    plt.rc('xtick.major', size=4, pad=4)
    plt.rc('xtick.minor', size=3, pad=4)
    plt.rc('ytick.major', size=4, pad=4)
    plt.rc('ytick.minor', size=3, pad=4)

    plt.rc('savefig',dpi=120)

    plt.legend(frameon=False)
    
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    # using aliases for color, linestyle and linewidth; gray, solid, thick
    #plt.rc('grid', c='0.5', ls='-', lw=5)
    plt.rc('lines', lw=1.5)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:44:15 2021

Wave packet dynamics solver for adiabatic dynamics with N vibrational modes
(N = 1 ,2)

For linear coordinates, use SPO method
For curvilinear coordinates, use RK4 method

@author: Bing Gu
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi
from numba import jit
from scipy.fftpack import fft2, ifft2, fftfreq
from numpy.linalg import inv, det



from lime.phys import rk4
from lime.units import au2fs

def APES0(x,y,z):
    """
    S0 surface of azobenzene
    """

    #########################################################################################
    ### S0 parameters                                                                     ###
    #########################################################################################

    E0  =          45724.2092465
    kt1  =          76901.9584118
    kt2  =          79689.8669574
    kt3  =          45756.7904038
    kt4  =          3191.20133757
    kt5  =          8416.87739554
    kt6  =         -16695.1731182
    k06  =           35.734941839
    k05  =         -607.288557767
    k04  =          4430.61165827
    k03  =          -17163.968902
    k02  =          38579.0137973
    k01  =         -54000.7844169
    k051  =          47.2264153087
    k042  =           30.685735969
    k033  =         -25.0925325083
    k041  =         -733.068559388
    k032  =      1.78059047023e-05
    k031  =           3754.6619712
    k022  =          242.897739415
    k021  =         -11088.2295672
    k011  =          14805.9595217
    kb16  =          65.5889623888
    kb15  =         -1129.87081212
    kb14  =          7788.63974022
    kb13  =         -29211.9914115
    kb12  =          64934.0540065
    kb11  =         -90116.5414213
    kc16  =           82.113866728
    kc15  =         -1374.00319054
    kc14  =          9663.63519695
    kc13  =         -35835.7804266
    kc12  =          75688.2185435
    kc11  =         -96498.0319856
    kd16  =          23.6479498282
    kd15  =         -514.266703221
    kd14  =           3972.6338256
    kd13  =         -15997.2379344
    kd12  =           37082.104648
    kd11  =         -53516.4198164
    ke16  =          14.8060327671
    ke15  =         -111.731924942
    ke14  =           274.78278865
    ke13  =          27.8538871862
    ke12  =         -211.292224507
    ke11  =         -496.434338735
    kf16  =         0.629498610196
    kf15  =         -79.2201576492
    kf14  =          695.784879089
    kf13  =         -2952.96315231
    kf12  =          6869.59564579
    kf11  =         -9863.05972704
    kg16  =         -24.1595997208
    kg15  =          386.420117106
    kg14  =         -2607.93005252
    kg13  =          9289.46355059
    kg12  =         -18520.6099137
    kg11  =          21648.8880939
    kb151  =          71.6435173825
    kb142  =         -51.1906065441
    kb133  =          25.3743063358
    kb141  =         -713.640679086
    kb132  =     -3.34212420423e-06
    kb131  =          4200.79161099
    kb122  =          480.053636589
    kb121  =         -15030.5163309
    kb111  =          22084.6055677
    kc151  =          70.6955203922
    kc142  =          43.4496502548
    kc133  =         -34.0759050735
    kc141  =         -1086.88895916
    kc132  =      1.24432203196e-05
    kc131  =          5552.74615021
    kc122  =          272.724611812
    kc121  =         -15941.8528919
    kc111  =          21288.6328075
    kd151  =          68.5007486211
    kd142  =         -39.9598957196
    kd133  =          22.4753597382
    kd141  =         -659.996101961
    kd132  =      2.66124434288e-06
    kd131  =          3450.25623761
    kd122  =          251.538835951
    kd121  =          -10658.921453
    kd111  =           14770.569593
    ke151  =         -50.3261510765
    ke142  =          30.2099904917
    ke133  =         -21.9753169762
    ke141  =          442.637598352
    ke132  =     -9.62019684443e-06
    ke131  =         -1924.21507821
    ke122  =          99.3465414777
    ke121  =          3347.65445618
    ke111  =         -2700.63441879
    kf151  =          30.2892302358
    kf142  =         -37.3961799766
    kf133  =          24.8498488454
    kf141  =         -185.269429314
    kf132  =      4.84474804947e-07
    kf131  =          881.929832424
    kf122  =          23.7574651812
    kf121  =         -2282.59027181
    kf111  =          2896.97002723
    kg151  =          -19.400390425
    kg142  =         -10.9316644589
    kg133  =            8.034858785
    kg141  =           291.09752732
    kg132  =     -8.14402503678e-06
    kg131  =         -1447.87143843
    kg122  =         -28.6364108643
    kg121  =          3728.33155078
    kg111  =         -4591.56198461

    return E0 + kt1*(cos(x/180*pi)) + kt2*(cos(2*x/180*pi)) + kt3*(cos(3*x/180*pi)) + kt4*(cos(4*x/180*pi)) + kt5*(cos(5*x/180*pi)) + kt6*(cos(6*x/180*pi))   \
  +     ( k06 + kb16*cos(x/180*pi) + kc16*cos(2*x/180*pi) + kd16*cos(3*x/180*pi) + ke16*cos(4*x/180*pi) + kf16*cos(5*x/180*pi) + kg16*cos(6*x/180*pi) ) * ( (y/180*pi)**6 + (z/180*pi)**6 )   +   ( k05 + kb15*cos(x/180*pi) + kc15*cos(2*x/180*pi) + kd15*cos(3*x/180*pi) + ke15*cos(4*x/180*pi) + kf15*cos(5*x/180*pi) + kg15*cos(6*x/180*pi) ) * ( (y/180*pi)**5 + (z/180*pi)**5 )   +   ( k04 + kb14*cos(x/180*pi) + kc14*cos(2*x/180*pi) + kd14*cos(3*x/180*pi) + ke14*cos(4*x/180*pi) + kf14*cos(5*x/180*pi) + kg14*cos(6*x/180*pi) ) * ( (y/180*pi)**4 + (z/180*pi)**4 )   +   ( k03 + kb13*cos(x/180*pi) + kc13*cos(2*x/180*pi) + kd13*cos(3*x/180*pi) + ke13*cos(4*x/180*pi) + kf13*cos(5*x/180*pi) + kg13*cos(6*x/180*pi) ) * ( (y/180*pi)**3 + (z/180*pi)**3 )   +   ( k02 + kb12*cos(x/180*pi) + kc12*cos(2*x/180*pi) + kd12*cos(3*x/180*pi) + ke12*cos(4*x/180*pi) + kf12*cos(5*x/180*pi) + kg12*cos(6*x/180*pi) ) * ( (y/180*pi)**2 + (z/180*pi)**2 )   +   ( k01 + kb11*cos(x/180*pi) + kc11*cos(2*x/180*pi) + kd11*cos(3*x/180*pi) + ke11*cos(4*x/180*pi) + kf11*cos(5*x/180*pi) + kg11*cos(6*x/180*pi) ) * ( (y/180*pi) + (z/180*pi) )   +   ( k051 + kb151*cos(x/180*pi) + kc151*cos(2*x/180*pi) + kd151*cos(3*x/180*pi) + ke151*cos(4*x/180*pi) + kf151*cos(5*x/180*pi) + kg151*cos(6*x/180*pi) ) * ( (y/180*pi)**5 * (z/180*pi) + (z/180*pi)**5 * (y/180*pi) )   +   ( k042 + kb142*cos(x/180*pi) + kc142*cos(2*x/180*pi) + kd142*cos(3*x/180*pi) + ke142*cos(4*x/180*pi) + kf142*cos(5*x/180*pi) + kg142*cos(6*x/180*pi) ) * ( (y/180*pi)**4 * (z/180*pi)**2 + (z/180*pi)**4 * (y/180*pi)**2 )   +   ( k033 + kb133*cos(x/180*pi) + kc133*cos(2*x/180*pi) + kd133*cos(3*x/180*pi) + ke133*cos(4*x/180*pi) + kf133*cos(5*x/180*pi) + kg133*cos(6*x/180*pi) ) * 2 * (y/180*pi)**3 * (z/180*pi)**3   +   ( k041 + kb141*cos(x/180*pi) + kc141*cos(2*x/180*pi) + kd141*cos(3*x/180*pi) + ke141*cos(4*x/180*pi) + kf141*cos(5*x/180*pi) + kg141*cos(6*x/180*pi) ) * ( (y/180*pi)**4 * (z/180*pi) + (z/180*pi)**4 * (y/180*pi) )   +   ( k032 + kb132*cos(x/180*pi) + kc132*cos(2*x/180*pi) + kd132*cos(3*x/180*pi) + ke132*cos(4*x/180*pi) + kf132*cos(5*x/180*pi) + kg132*cos(6*x/180*pi) ) * ( (y/180*pi)**3 * (z/180*pi)**2 + (z/180*pi)**3 * (y/180*pi) )**2  +   ( k031 + kb131*cos(x/180*pi) + kc131*cos(2*x/180*pi) + kd131*cos(3*x/180*pi) + ke131*cos(4*x/180*pi) + kf131*cos(5*x/180*pi) + kg131*cos(6*x/180*pi) ) * ( (y/180*pi)**3 * (z/180*pi) + (z/180*pi)**3 * (y/180*pi) )   +   ( k022 + kb122*cos(x/180*pi) + kc122*cos(2*x/180*pi) + kd122*cos(3*x/180*pi) + ke122*cos(4*x/180*pi) + kf122*cos(5*x/180*pi) + kg122*cos(6*x/180*pi) ) * 2 * (y/180*pi)**2 * (z/180*pi)**2   +   ( k021 + kb121*cos(x/180*pi) + kc121*cos(2*x/180*pi) + kd121*cos(3*x/180*pi) + ke121*cos(4*x/180*pi) + kf121*cos(5*x/180*pi) + kg121*cos(6*x/180*pi) ) * ( (y/180*pi)**2 * (z/180*pi) + (z/180*pi)**2 * (y/180*pi) )   +   ( k011 + kb111*cos(x/180*pi) + kc111*cos(2*x/180*pi) + kd111*cos(3*x/180*pi) + ke111*cos(4*x/180*pi) + kf111*cos(5*x/180*pi) + kg111*cos(6*x/180*pi) ) * 2 * (y/180*pi) * (z/180*pi)



def APES1(x, y, z):
    """
    S1 APES
    """
    E0  =          16535.7014697
    kt1  =          10015.4335703
    kt2  =          19203.2136601
    kt3  =          14712.3191617
    kt4  =          32593.7148677
    kt5  =         -14347.7137229
    kt6  =         -9608.19664088
    k06  =          27.6620432179
    k05  =         -337.417577592
    k04  =          1629.31572625
    k03  =         -4635.90139339
    k02  =          10124.4188861
    k01  =         -14199.6859032
    k051  =         -46.8909784782
    k042  =         -30.0941411313
    k033  =          16.3321999846
    k041  =          640.393504426
    k032  =     -2.08345552839e-05
    k031  =         -2619.58225413
    k022  =          236.906019428
    k021  =          3570.68782547
    k011  =         -1836.04051302
    kb16  =         -72.7295084028
    kb15  =          917.073547882
    kb14  =         -4795.74171044
    kb13  =          12344.8822433
    kb12  =         -13324.7662415
    kb11  =         -6547.16959566
    kc16  =        -0.900678750999
    kc15  =          169.849475859
    kc14  =         -1155.34786092
    kc13  =          2902.66879583
    kc12  =          979.022137137
    kc11  =         -16862.5585023
    kd16  =          14.0470095478
    kd15  =         -208.248258129
    kd14  =          1087.27567628
    kd13  =         -3545.46377377
    kd12  =          8696.70868056
    kd11  =         -14477.6299922
    ke16  =          51.0683797301
    ke15  =         -804.431546713
    ke14  =           5195.7230133
    ke13  =          -17808.507884
    ke12  =          34896.2568892
    ke11  =         -39726.6350967
    kf16  =          21.1751944633
    kf15  =         -153.500032295
    kf14  =          3.11931693623
    kf13  =          2906.73307073
    kf12  =         -10298.8102751
    kf11  =          19124.1210006
    kg16  =          5.98763617613
    kg15  =          7.43250735452
    kg14  =         -392.487520388
    kg13  =          2441.46643782
    kg12  =         -6980.44306194
    kg11  =          11417.6380067
    kb151  =          50.8493289907
    kb142  =         -33.8097823817
    kb133  =          16.4587915193
    kb141  =         -473.741145251
    kb132  =     -9.99896774519e-06
    kb131  =          2628.25961567
    kb122  =          350.199793334
    kb121  =         -9528.56046798
    kb111  =          14065.1365483
    kc151  =         -69.8247171493
    kc142  =           111.03252646
    kc133  =         -82.5418396219
    kc141  =          224.822446444
    kc132  =     -1.13721077853e-05
    kc131  =         -293.110687698
    kc122  =          419.786382886
    kc121  =          -3990.3594669
    kc111  =          8861.20215057
    kd151  =         -1.10530938178
    kd142  =         -58.0146484969
    kd133  =           36.091848212
    kd141  =          235.171836934
    kd132  =     -9.04956949168e-06
    kd131  =         -761.610190416
    kd122  =          171.444222977
    kd121  =         -259.241104953
    kd111  =          2252.48164276
    ke151  =          22.4827656096
    ke142  =         -5.43217580494
    ke133  =          2.05746204115
    ke141  =         -245.759102376
    ke132  =        5.476902422e-06
    ke131  =          1230.89403186
    ke122  =          109.618135107
    ke121  =         -3838.20730051
    ke111  =          5320.24135964
    kf151  =         -66.9493923062
    kf142  =          2.45901101922
    kf133  =           -1.170939241
    kf141  =          796.027825643
    kf132  =      -9.3111496105e-06
    kf131  =          -3806.9193028
    kf122  =          -20.796813226
    kf121  =          9090.84495619
    kf111  =         -10594.9929306
    kg151  =         -32.9769364379
    kg142  =          43.7576758902
    kg133  =         -27.9930199051
    kg141  =          199.762883086
    kg132  =     -1.35781806673e-06
    kg131  =         -1025.24754351
    kg122  =         -78.7711729666
    kg121  =          3069.87461383
    kg111  =         -4042.09772539

    return E0 + kt1*(cos(x/180*pi)) + kt2*(cos(2*x/180*pi)) + kt3*(cos(3*x/180*pi)) \
    + kt4*(cos(4*x/180*pi)) + kt5*(cos(5*x/180*pi)) + kt6*(cos(6*x/180*pi))     +\
     ( k06 + kb16*cos(x/180*pi) + kc16*cos(2*x/180*pi) + kd16*cos(3*x/180*pi) + \
      ke16*cos(4*x/180*pi) + kf16*cos(5*x/180*pi) + kg16*cos(6*x/180*pi) ) * \
      ( (y/180*pi)**6 + (z/180*pi)**6 )   +   ( k05 + kb15*cos(x/180*pi) + kc15*cos(2*x/180*pi) \
       + kd15*cos(3*x/180*pi) + ke15*cos(4*x/180*pi) + kf15*cos(5*x/180*pi) \
       + kg15*cos(6*x/180*pi) ) * ( (y/180*pi)**5 + (z/180*pi)**5 )   \
       +   ( k04 + kb14*cos(x/180*pi) + kc14*cos(2*x/180*pi) + kd14*cos(3*x/180*pi) \
    + ke14*cos(4*x/180*pi) + kf14*cos(5*x/180*pi) + kg14*cos(6*x/180*pi) ) * ( (y/180*pi)**4 \
       + (z/180*pi)**4 )   +   ( k03 + kb13*cos(x/180*pi) + kc13*cos(2*x/180*pi) \
    + kd13*cos(3*x/180*pi) + ke13*cos(4*x/180*pi) + kf13*cos(5*x/180*pi) + \
    kg13*cos(6*x/180*pi) ) * ( (y/180*pi)**3 + (z/180*pi)**3 )   \
    +   ( k02 + kb12*cos(x/180*pi) + kc12*cos(2*x/180*pi) + kd12*cos(3*x/180*pi) \
    + ke12*cos(4*x/180*pi) + kf12*cos(5*x/180*pi) + kg12*cos(6*x/180*pi) ) * ( (y/180*pi)**2 + (z/180*pi)**2 )   +   ( k01 + kb11*cos(x/180*pi) + kc11*cos(2*x/180*pi) + kd11*cos(3*x/180*pi) + ke11*cos(4*x/180*pi) + kf11*cos(5*x/180*pi) + kg11*cos(6*x/180*pi) ) * ( (y/180*pi) + (z/180*pi) )   +   ( k051 + kb151*cos(x/180*pi) + kc151*cos(2*x/180*pi) + kd151*cos(3*x/180*pi) + ke151*cos(4*x/180*pi) + kf151*cos(5*x/180*pi) + kg151*cos(6*x/180*pi) ) * ( (y/180*pi)**5 * (z/180*pi) + (z/180*pi)**5 * (y/180*pi) )   +   ( k042 + kb142*cos(x/180*pi) + kc142*cos(2*x/180*pi) + kd142*cos(3*x/180*pi) + ke142*cos(4*x/180*pi) + kf142*cos(5*x/180*pi) + kg142*cos(6*x/180*pi) ) * ( (y/180*pi)**4 * (z/180*pi)**2 + (z/180*pi)**4 * (y/180*pi)**2 )   +   ( k033 + kb133*cos(x/180*pi) + kc133*cos(2*x/180*pi) + kd133*cos(3*x/180*pi) + ke133*cos(4*x/180*pi) + kf133*cos(5*x/180*pi) + kg133*cos(6*x/180*pi) ) * 2 * (y/180*pi)**3 * (z/180*pi)**3   +   ( k041 + kb141*cos(x/180*pi) + kc141*cos(2*x/180*pi) + kd141*cos(3*x/180*pi) + ke141*cos(4*x/180*pi) + kf141*cos(5*x/180*pi) + kg141*cos(6*x/180*pi) ) * ( (y/180*pi)**4 * (z/180*pi) + (z/180*pi)**4 * (y/180*pi) )   +   ( k032 + kb132*cos(x/180*pi) + kc132*cos(2*x/180*pi) + kd132*cos(3*x/180*pi) + ke132*cos(4*x/180*pi) + kf132*cos(5*x/180*pi) + kg132*cos(6*x/180*pi) ) * ( (y/180*pi)**3 * (z/180*pi)**2 + (z/180*pi)**3 * (y/180*pi) )**2  +   ( k031 + kb131*cos(x/180*pi) + kc131*cos(2*x/180*pi) + kd131*cos(3*x/180*pi) + ke131*cos(4*x/180*pi) + kf131*cos(5*x/180*pi) + kg131*cos(6*x/180*pi) ) * ( (y/180*pi)**3 * (z/180*pi) + (z/180*pi)**3 * (y/180*pi) )   +   ( k022 + kb122*cos(x/180*pi) + kc122*cos(2*x/180*pi) + kd122*cos(3*x/180*pi) + ke122*cos(4*x/180*pi) + kf122*cos(5*x/180*pi) + kg122*cos(6*x/180*pi) ) * 2 * (y/180*pi)**2 * (z/180*pi)**2   +   ( k021 + kb121*cos(x/180*pi) + kc121*cos(2*x/180*pi) + kd121*cos(3*x/180*pi) + ke121*cos(4*x/180*pi) + kf121*cos(5*x/180*pi) + kg121*cos(6*x/180*pi) ) * ( (y/180*pi)**2 * (z/180*pi) + (z/180*pi)**2 * (y/180*pi) )   +   ( k011 + kb111*cos(x/180*pi) + kc111*cos(2*x/180*pi) + kd111*cos(3*x/180*pi) + ke111*cos(4*x/180*pi) + kf111*cos(5*x/180*pi) + kg111*cos(6*x/180*pi) ) * 2 * (y/180*pi) * (z/180*pi)

@jit
def gauss_x_2d(sigma, x0, y0, kx0, ky0):
    """
    generate the gaussian distribution in 2D grid
    :param x0: float, mean value of gaussian wavepacket along x
    :param y0: float, mean value of gaussian wavepacket along y
    :param sigma: float array, covariance matrix with 2X2 dimension
    :param kx0: float, initial momentum along x
    :param ky0: float, initial momentum along y
    :return: gauss_2d: float array, the gaussian distribution in 2D grid
    """
    gauss_2d = np.zeros((len(x), len(y)), dtype=np.complex128)

    for i in range(len(x)):
        for j in range(len(y)):
            delta = np.dot(np.array([x[i]-x0, y[j]-y0]), inv(sigma))\
                      .dot(np.array([x[i]-x0, y[j]-y0]))
            gauss_2d[i, j] = (np.sqrt(det(sigma))
                              * np.sqrt(np.pi) ** 2) ** (-0.5) \
                              * np.exp(-0.5 * delta + 1j
                                       * np.dot(np.array([x[i], y[j]]),
                                                  np.array([kx0, ky0])))

    return gauss_2d


@jit
def potential_2d(x_range_half, y_range_half, couple_strength, couple_type):
    """
    generate two symmetric harmonic potentials wrt the origin point in 2D
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return: v_2d: float list, a list containing for matrices:
                               v_2d[0]: the first potential matrix
                               v_2d[1]: the potential coupling matrix
                                        between the first and second
                               v_2d[2]: the potential coupling matrix
                                        between the second and first
                               v_2d[3]: the second potential matrix
    """
    v_2d = [0, 0, 0, 0]
    v_2d[0] = (xv + x_range_half) ** 2 / 2.0 + (yv + y_range_half) ** 2 / 2.0
    v_2d[3] = (xv - x_range_half) ** 2 / 2.0 + (yv - y_range_half) ** 2 / 2.0

    # x_cross = sympy.Symbol('x_cross')
    # mu = sympy.solvers.solve(
    #     (x_cross - x_range_half) ** 2 / 2.0 -
    #     (x_cross + x_range_half) ** 2 / 2.0,
    #     x_cross)

    if couple_type == 0:
        v_2d[1] = np.zeros(np.shape(v_2d[0]))
        v_2d[2] = np.zeros(np.shape(v_2d[0]))
    elif couple_type == 1:
        v_2d[1] = np.full((np.shape(v_2d[0])), couple_strength)
        v_2d[2] = np.full((np.shape(v_2d[0])), couple_strength)
    elif couple_type == 2:
        v_2d[1] = couple_strength * (xv+yv)
        v_2d[2] = couple_strength * (xv+yv)
    # elif couple_type == 3:
    #     v_2d[1] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    #     v_2d[2] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    else:
        raise 'error: coupling type not existing'

    return v_2d


@jit
def diabatic(x, y):
    """
    PESs in diabatic representation
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return:
        v:  float 2d array, matrix elements of the DPES and couplings
    """
    nstates = 2

    v = np.zeros((nstates, nstates))

    v[0,0] = (x + 4.) ** 2 / 2.0 + (y + 3.) ** 2 / 2.0
    v[1,1] = (x - 4.) ** 2 / 2.0 + (y - 3.) ** 2 / 2.0

    v[0, 1] = v[1, 0] = 0

    return v

# @jit
# def x_evolve_half_2d(dt, v_2d, psi_grid):
#     """
#     propagate the state in grid basis half time step forward with H = V
#     :param dt: float
#                 time step
#     :param v_2d: float array
#                 the two electronic states potential operator in grid basis
#     :param psi_grid: list
#                 the two-electronic-states vibrational state in grid basis
#     :return: psi_grid(update): list
#                 the two-electronic-states vibrational state in grid basis
#                 after being half time step forward
#     """

#     for i in range(len(x)):
#         for j in range(len(y)):
#             v_mat = np.array([[v_2d[0][i, j], v_2d[1][i, j]],
#                              [v_2d[2][i, j], v_2d[3][i, j]]])

#             w, u = scipy.linalg.eigh(v_mat)
#             v = np.diagflat(np.exp(-0.5 * 1j * w / hbar * dt))
#             array_tmp = np.array([psi_grid[0][i, j], psi_grid[1][i, j]])
#             array_tmp = np.dot(u.conj().T, v.dot(u)).dot(array_tmp)
#             psi_grid[0][i, j] = array_tmp[0]
#             psi_grid[1][i, j] = array_tmp[1]
#             #self.x_evolve = self.x_evolve_half * self.x_evolve_half
#             #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
#             #               (self.k * self.k) * dt)


@jit
def x_evolve_2d(dt, psi, v):
    """
    propagate the state in grid basis half time step forward with H = V
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = np.exp(- 1j * v * dt) * psi


    return vpsi


def k_evolve_2d(dt, kx, ky, psi):
    """
    propagate the state in grid basis a time step forward with H = K
    :param dt: float, time step
    :param kx: float, momentum corresponding to x
    :param ky: float, momentum corresponding to y
    :param psi_grid: list, the two-electronic-states vibrational states in
                           grid basis
    :return: psi_grid(update): list, the two-electronic-states vibrational
                                     states in grid basis
    """

    psi_k = fft2(psi)
    mx, my = mass

    Kx, Ky = np.meshgrid(kx, ky)

    kin = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

    psi_k = kin * psi_k
    psi = ifft2(psi_k)

    return psi


def dpsi(psi, kx, ky, ndim=2):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.
    ndim : int, default 2
        coordinates dimension
    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    kpsi = np.zeros((nx, ny, ndim), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

    return kpsi

def dxpsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi

def dypsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi


def adiabatic_2d(x, y, psi0, v, dt, Nt=0, coords='linear', mass=None, G=None):
    """
    propagate the adiabatic dynamics at a single surface

    :param dt: time step
    :param v: 2d array
                potential matrices in 2D
    :param psi: list
                the initial state
    mass: list of 2 elements
        reduced mass

    Nt: int
        the number of the time steps, Nt=0 indicates that no propagation has been done,
                   only the initial state and the initial purity would be
                   the output

    G: 4D array nx, ny, ndim, ndim
        G-matrix

    :return: psi_end: list
                      the final state

    G: 2d array
        G matrix only used for curvilinear coordinates
    """
    #f = open('density_matrix.dat', 'w')
    t = 0.0
    dt2 = dt * 0.5

    psi = psi0.copy()

    nx, ny = psi.shape

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    if coords == 'linear':
        # Split-operator method for linear coordinates

        psi = x_evolve_2d(dt2, psi,v)

        for i in range(Nt):
            t += dt
            psi = k_evolve_2d(dt, kx, ky, psi)
            psi = x_evolve_2d(dt, psi, v)

    elif coords == 'curvilinear':

        # kxpsi = np.einsum('i, ijn -> ijn', kx, psi_k)
        # kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

        # tpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dypsi = np.zeros((nx, ny, nstates), dtype=complex)

        # for i in range(nstates):

        #     dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
        #     dypsi[:,:,i] = ifft2(kypsi[:,:,i])

        for k in range(Nt):
            t += dt
            psi = rk4(psi, hpsi, dt, kx, ky, v, G)

        #f.write('{} {} {} {} {} \n'.format(t, *rho))
        #purity[i] = output_tmp[4]



    # t += dt
    #f.close()

    return psi

def KEO(psi, kx, ky, G):
    '''
    compute kinetic energy operator K * psi

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
#    kpsi = dpsi(psi, kx, ky)

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    nx, ny = len(kx), len(ky)
    kpsi = np.zeros((nx, ny, 2), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

#   ax.contour(x, y, np.abs(kpsi[:,:,1]))

    tmp = np.einsum('ijrs, ijs -> ijr', G, kpsi)
    #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

    # Fourier transform of the wavefunction
    phi_x = tmp[:,:,0]
    phi_y = tmp[:,:,1]

    phix_k = fft2(phi_x)
    phiy_k = fft2(phi_y)

    # momentum operator in the Fourier space
    kxphi = np.einsum('i, ij -> ij', kx, phix_k)
    kyphi = np.einsum('j, ij -> ij', ky, phiy_k)

    # transform back to coordinate space
    kxphi = ifft2(kxphi)
    kyphi = ifft2(kyphi)

    # psi += -1j * dt * 0.5 * (kxphi + kyphi)

    return 0.5 * (kxphi + kyphi)

def PEO(psi, v):
    """
    V |psi>
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = v * psi
    return vpsi

def hpsi(psi, kx, ky, v, G):

    kpsi = KEO(psi, kx, ky, G)
    vpsi = PEO(psi, v)

    return -1j * (kpsi + vpsi)

######################################################################
# Helper functions for gaussian wave-packets


def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

@jit
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y


def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))

@jit
def density_matrix(psi_grid):
    """
    compute electronic purity from the wavefunction
    """
    rho00 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[0]))*dx*dy
    rho01 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[1]))*dx*dy
    rho11 = np.sum(np.multiply(np.conj(psi_grid[1]), psi_grid[1]))*dx*dy

    purity = rho00**2 + 2*rho01*rho01.conj() + rho11**2

    return rho00, rho01, rho01.conj(), rho11, purity



if __name__ == '__main__':

    # specify time steps and duration
    ndim = 2 # 2D problem, DO NOT CHANGE!
    dt = 0.01
    print('time step = {} fs'.format(dt * au2fs))

    num_steps = 2000


    nx = 2 ** 6
    ny = 2 ** 6
    xmin = -8
    xmax = -xmin
    ymin = -8
    ymax = -ymin
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # k-space grid
    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    v = 0.5 * (X**2 + Y**2)

    # for i in range(nx):
    #     for j in range(ny):
    #         v[i,j] = diabatic(x[i], y[j])[0,0]

    #ax.imshow(v)

    # specify constants
    mass = [1.0, 1.0]  # particle mass

    x0, y0, kx0, ky0 = -3, -1, 2.0, 0

    #coeff1, phase = np.sqrt(0.5), 0

    print('x range = ', x[0], x[-1])
    print('dx = {}'.format(dx))
    print('number of grid points along x = {}'.format(nx))
    print('y range = ', y[0], y[-1])
    print('dy = {}'.format(dy))
    print('number of grid points along y = {}'.format(ny))

    sigma = np.identity(2) * 0.5

    psi0 =   gauss_x_2d(sigma, x0, y0, kx0, ky0)

    fig, ax = plt.subplots()
    ax.contour(x, y, np.abs(psi0).T)

    #psi = psi0

    # propagate

    # store the final wavefunction
    #f = open('wft.dat','w')
    #for i in range(N):
    #    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    #f.close()


    G = np.zeros((nx, ny, ndim, ndim))
    G[:,:,0, 0] = G[:,:,1, 1] = 1.

    fig, ax = plt.subplots()
    extent=[xmin, xmax, ymin, ymax]

    psi1 = adiabatic_2d(x, y, psi0, v, dt=dt, Nt=num_steps, coords='curvilinear',G=G)
    ax.contour(x,y, np.abs(psi1).T)

    fig, ax = plt.subplots()

    psi2 = adiabatic_2d(psi0, v, mass=mass, dt=dt, Nt=num_steps)
    ax.contour(x,y, np.abs(psi2).T)



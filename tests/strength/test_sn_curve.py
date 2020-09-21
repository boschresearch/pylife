# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 07:28:32 2020

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
from pylife.strength import sn_curve


def test_calc_damage():
   
    nCode_Xbinsize = 62.375
    nCode_XMax = 468.8125
    nCode_XMin = 32.1875
    material = pd.DataFrame(index = ['k_1', 'ND_50', 'SD_50'],
                                  columns = ['elementar','MinerHaibach','original'],
                                  data = [[4,5,6],[4e7,1e6,1e8],[200,180,150]])
    
    index =  pd.interval_range(start=nCode_XMin-nCode_Xbinsize/2, 
                               end = nCode_XMax+nCode_Xbinsize/2, periods=8,name = "range")
    loads = pd.DataFrame([[1.227E5,1.114E5,1.829E5],[2.433E4,3.117E4,16],
                          [1591,5095,0],[178,427,0],[64,138,0],[8,63,0],
                          [0,24,0],[0,1,0]],
                               index = index)
    nCode_damage_elementar = pd.DataFrame([[2.057849E-06,1.868333E-06,3.067487E-06],
                                            [3.039750E-05,3.894329E-05,1.999014E-08],
                                            [1.507985E-05,4.829155E-05,0.000000E+00],
                                            [6.434856E-06,1.543643E-05,0.000000E+00],
                                            [6.296737E-06,1.357734E-05,0.000000E+00],
                                            [1.751881E-06,1.379607E-05,0.000000E+00],
                                            [0.000000E+00,1.023415E-05,0.000000E+00],
                                            [0.000000E+00,7.548524E-07,0.000000E+00]],
                                index = index)
    nCode_damage_MinerHaibach = pd.DataFrame([[2.293973E-08,2.083222E-08,3.419927E-08],
                                              [7.414563E-05,9.499395E-05,4.876799E-08],
                                              [4.631857E-04,1.483300E-03,0.000000E+00],
                                              [4.779392E-04,1.146517E-03,0.000000E+00],
                                              [6.006944E-04,1.295247E-03,0.000000E+00],
                                              [2.041326E-04,1.607545E-03,0.000000E+00],
                                              [0.000000E+00,1.408691E-03,0.000000E+00],
                                              [0.000000E+00,1.198482E-04,0.000000E+00]],
                                             index = index)
    nCode_damage_original = pd.DataFrame([[0.00000000E+00,0.00000000E+00,0.00000000E+00],
                                          [0.00000000E+00,0.00000000E+00,0.00000000E+00],
                                          [2.08681840E-05,6.68280320E-05,0.00000000E+00],
                                          [1.73881920E-05,4.17121220E-05,0.00000000E+00],
                                          [2.80698130E-05,6.05255350E-05,0.00000000E+00],
                                          [1.16511320E-05,9.17526660E-05,0.00000000E+00],
                                          [0.00000000E+00,9.49790820E-05,0.00000000E+00],
                                          [0.00000000E+00,9.32071420E-06,0.00000000E+00]],
                                             index = index)
    damage_nCode = [nCode_damage_elementar,nCode_damage_MinerHaibach,nCode_damage_original]
    ii = 0
    damage_list = []
    for method in material:
        damage_calc = sn_curve.FiniteLifeCurve(**material[method])
        damage = damage_calc.calc_damage(loads, method=method)
        pd.testing.assert_frame_equal(damage, damage_nCode[ii], rtol=0.001, atol=0)
        ii += 1

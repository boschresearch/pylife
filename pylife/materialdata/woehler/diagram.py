# Copyright (c) 2019 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Mustapha Kassem"
__maintainer__ = "Johannes Mueller"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


class PlotWoehlerCurve:

    def __init__(self, WoehlerCurve, title, method, amp, ld_typ, unit, xlim_WL, ylim_WL, default_diag):
        if default_diag == 1:
            self.amp = u'Amplitude'
            # Load or Stress
            self.ld_typ = u'Stress'
            # Unit
            self.unit = u'$N/mm^2$'
            # Figure xy limits (Woehler curve)
            self.xlim_WL = (round(min(WoehlerCurve.fatigue_data.cycles)*0.4,-1), round(max(WoehlerCurve.fatigue_data.cycles)*2,-1))
            self.ylim_WL = (round(min(WoehlerCurve.fatigue_data.loads)*0.8,-1), round(max(WoehlerCurve.fatigue_data.loads)*1.2,-1))
        else:
            # Amplitude
            self.amp = amp
            # Load or Stress
            self.ld_typ = ld_typ
            # Unit
            self.unit = unit
            # Figure xy limits (Woehler curve)
            self.xlim_WL = xlim_WL
            self.ylim_WL = ylim_WL

        if title.startswith('Probability plot'):
            self.probability_plot(WoehlerCurve, title)
        elif title == 'Plot':
            self.calc_woehlercurve_from_curve(WoehlerCurve, method)
        else:
            self.calc_woehlercurve_from_curve(WoehlerCurve, method)
            '''
            if title.startswith('Probability plot of the finite zone'):
                self.probability_plot(WoehlerCurve, title)
            else:
            '''
            self.base_plot(WoehlerCurve, title, method)
            self.edit_WL_diagram_from_curve()

    def shift_woehlercurve_pf(self, WL50, WoehlerCurve, pa_goal, method):
        """ Shift the Basquin-curve according to the failure probability value (obtain the 10-90 % curves)"""
        
        TN = WoehlerCurve.TN

        WL_shift = np.array(WL50)
        WL_shift[:, 1] /= (10**(-stats.norm.ppf(pa_goal)*np.log10(TN)/2.56))

        return WL_shift


    def base_plot(self, WoehlerCurve, title, method):
        k = WoehlerCurve.k
        TN = WoehlerCurve.TN
        if method == 'Mali':
            TS = TN**(1./k)
        else:
            TS = WoehlerCurve.TS

        self.fig = plt.figure(figsize=(8, 5))
        if title == 'Initial data':
            plt.plot(WoehlerCurve.fatigue_data.fractures.cycles, WoehlerCurve.fatigue_data.fractures.loads, 'bo', label='Failure')
            plt.axhline(y=WoehlerCurve.fatigue_data.fatg_lim, linewidth=2, color='r', label='Endurance limit')
        else:
            self.ax = plt.subplot('111')
            plt.plot(WoehlerCurve.fatigue_data.fractures.cycles, WoehlerCurve.fatigue_data.fractures.loads, 'bo', label='Failure')
            plt.plot(self.wl_curve[:,1], self.wl_curve[:,0], 'r', linewidth=2., label=u'WL, $P_A$=50%')

            if title == 'Slope':
                text = '$k$ = '+str(np.round(k, decimals=2))
                plt.text(0.01, 0.03, text, verticalalignment='bottom',
                         horizontalalignment='left', transform=self.ax.transAxes,
                         bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})

            elif title == 'Pearl chain method':
                plt.plot(WoehlerCurve.fatigue_data.N_shift, np.ones(len(WoehlerCurve.fatigue_data.N_shift))*WoehlerCurve.fatigue_data.Sa_shift,
                         'go',label='PCM shifted probes', marker="v")
                plt.plot(self.xlim_WL, np.ones(len(self.xlim_WL))*WoehlerCurve.fatigue_data.Sa_shift,'g')

            elif title == 'Deviation TN':
                plt.plot(self.shift_woehlercurve_pf(WL50=self.wl_curve,
                                                                WoehlerCurve=WoehlerCurve, pa_goal=0.1,
                                                                method=method)[:, 1],
                        self.wl_curve[:, 0], 'r', linewidth=1.5,
                        linestyle='--', label=u'WL, $P_A$=10% u. 90%'
                        )
                plt.plot(self.shift_woehlercurve_pf(WL50=self.wl_curve,
                                                                WoehlerCurve=WoehlerCurve, pa_goal=0.9,
                                                                method=method)[:, 1],
                         self.wl_curve[:, 0], 'r', linewidth=1.5, linestyle='--'
                         )

                text = '$k$ = '+str(np.round(k, decimals=2)) + '\n'
                text += '$1/T_N$ = ' + str(np.round(TN, decimals=2)) + '\n'
                text += '$1/T_S^*$ = ' + str(np.round(TS, decimals=2))

                plt.text(0.01, 0.03, text, verticalalignment='bottom',
                         horizontalalignment='left', transform=self.ax.transAxes,
                         bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})

        plt.plot(WoehlerCurve.fatigue_data.runouts.cycles, WoehlerCurve.fatigue_data.runouts.loads, 'bo', mfc='none',
                 label=u'Runout', alpha=0.5
                 )
        plt.title(title)

        return self.fig


    def edit_WL_diagram_from_curve(self):
        """ Transforming the axis to accommodate the data in the logrithmic scale """
        self.edit_WL_diagram(self.fig, self.amp, self.ld_typ, self.unit, self.xlim_WL, self.ylim_WL)
         
    def edit_WL_diagram(self, figure, amp, ld_typ, unit, xlim='auto', ylim='auto'):
        """ Transforming the axis to accommodate the data in the logrithmic scale """
        if xlim != 'auto':
            plt.xlim(xlim)
        if ylim != 'auto':
            plt.ylim(ylim)

        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.xlabel('Number of cycles')
        plt.ylabel(amp+' ('+ld_typ+') in '+unit+' (log scaled)')
        plt.legend(loc='upper right', fontsize=12)
        figure.tight_layout()
        matplotlib.rcParams.update({'font.size': 12})        


    def probability_plot(self, WoehlerCurve, title):
        if title == 'Probability plot of the finite zone':
            X = WoehlerCurve.fatigue_data.N_shift
            Y = WoehlerCurve.u
            a = WoehlerCurve.a_pa
            b = WoehlerCurve.b_pa
            T = WoehlerCurve.TN
            xlab = 'load cycle N'
            scatter = '$1/T_N$ = '

        elif title == 'Probability plot of the infinite zone':
            X = WoehlerCurve.ld_lvls_inf[0]
            Y = WoehlerCurve.inv_cdf
            a = WoehlerCurve.a_ue
            b = WoehlerCurve.b_ue
            T = WoehlerCurve.Probit_result['1/TS']
            xlab = self.amp+' ('+self.ld_typ+') in '+self.unit
            scatter = '$1/T_S$ = '

        self.fig = plt.figure(figsize=(6, 4))
        self.ax = plt.subplot('111')


        plt.plot(X, Y, 'ro')
        plt.plot([10**((i-b)/a) for i in np.arange(-2.5, 2.5, 0.1)],
                 np.arange(-2.5, 2.5, 0.1), 'r')

        yticks = [1, 5, 10, 20, 50, 80, 90, 95, 99]
        plt.yticks([stats.norm.ppf(i/100.) for i in yticks],
                   [str(i)+' %' for i in yticks])

        plt.xticks([10**((stats.norm.ppf(0.1)-b)/a), 10**((stats.norm.ppf(0.9)-b)/a)],
                    ('', ''))

        plt.xscale('log') #problem with the scaling cant overwrite xticks for inf zone
        plt.grid()
        plt.xlabel(xlab)
        plt.ylabel('Failure probability')
        plt.title(title)

        plt.text(0.15, 0.03, 'N($P_{A,10}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.1)-b)/a), decimals=1),
                 verticalalignment='bottom',horizontalalignment='left', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)

        plt.text(0.5, 0.88, scatter + str(np.round(T,decimals=2)),
                 verticalalignment='bottom',horizontalalignment='center', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10}, fontsize=11)

        plt.text(0.9, 0.03, 'N($P_{A,90}$)='+'{:1.1e}'.format(10**((stats.norm.ppf(0.9)-b)/a),decimals=1),
                 verticalalignment='bottom',horizontalalignment='right', transform=self.ax.transAxes,
                 bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10}, fontsize=11)

        if title == 'Probability plot of the finite zone':
            plt.xticks([10**((stats.norm.ppf(0.1)-b)/a), 10**((stats.norm.ppf(0.9)-b)/a)],
                        ('', ''))
        else:
            plt.xticks([10**((stats.norm.ppf(0.1)-b)/a), 10**((stats.norm.ppf(0.9)-b)/a)],
                       ('', ''))
            #('N($P_{A,10}$)', 'N($P_{A,90}$)'))

        self.fig.tight_layout()


    def runout_zone_method(self, WoehlerCurve, method, slope_chosen):

        #print('\n------ Method for runout-zone -------')
        if method == 'Mali':
            SD50 = WoehlerCurve.Mali_5p_result['SD_50']
            ND50 = WoehlerCurve.Mali_5p_result['ND_50']
            TS = WoehlerCurve.Mali_5p_result['1/TS']
            k_1 = WoehlerCurve.Mali_5p_result['k_1']
            #print('\nMethod chosen: Maximum Likelihood')

        else:
            SD50 = WoehlerCurve.Probit_result['SD_50']
            ND50 = WoehlerCurve.Probit_result['ND_50']
            TS = WoehlerCurve.Probit_result['1/TS']
            k_1 = WoehlerCurve.k
            #print('\nMethod chosen: Probit')

        print('\n------ K_2 -------')
        if slope_chosen == 0:
            print('\nBasic palmgren-Miner rule: k_2 = 0')
        elif slope_chosen == k_1:
            print('\nElementary palmgern-Miner rule: k_2 = k_1')
        elif slope_chosen == (2*k_1)-1:
            print('\nHaibach rule: k_2 = 2 * k_1 - 1')
        else:
            print('\nk_2 is estimated with a linear regression function')

        return SD50, ND50, TS


    def final_curve_plot(self, WoehlerCurve, SD50, ND50, TS, slope, method,
                         amp, ld_typ, unit, xlim_WL, ylim_WL, default_diag):
        WC_data = WoehlerCurve
        if default_diag == 1:
            amp = u'Amplitude'
            # Load or Stress
            ld_typ = u'Stress'
            # Unit
            unit = u'$N/mm^2$'
            # Figure xy limits (Woehler curve)
            xlim_WL = (round(min(WC_data.data.cycles)*0.4,-1), round(max(WC_data.data.cycles)*2,-1))
            ylim_WL = (round(min(WC_data.data.loads)*0.8,-1), round(max(WC_data.data.loads)*1.2,-1))

        else:
            # Amplitude
            amp = amp
            # Load or Stress
            ld_typ = ld_typ
            # Unit
            unit = unit
            # Figure xy limits (Woehler curve)
            xlim_WL = xlim_WL
            ylim_WL = ylim_WL

        # Method
        if method == 'Mali':
            k_1 = WC_data.Mali_5p_result['k_1']
            TN = WC_data.Mali_5p_result['1/TN']
        else:
            k_1 = WC_data.k
            TN = WC_data.TN

        fig = plt.figure(figsize=(8, 5))
        #fig = plt.figure(figsize=(4.5,5.5))
        ax = plt.subplot('111')

        plt.plot(WC_data.fractures.cycles, WC_data.fractures.loads, 'bo', label='Failure')
        plt.plot(WC_data.runouts.cycles, WC_data.runouts.loads, 'bo', mfc='none',
                 label=u'Runout')


        WL_50 = self.calc_woehlercurve(k=k_1, N0=ND50, S0=SD50, y_min=SD50, y_max=WC_data.loads_max*1.2)

        # Werte für Wöhlerlinie
        SD10 = SD50 / (10**(-stats.norm.ppf(0.1)*np.log10(TS)/2.56))
        SD90 = SD50 / (10**(-stats.norm.ppf(0.9)*np.log10(TS)/2.56))

        if slope == 0:
            WL_50 = np.append(WL_50, np.array([[SD50, 1E9]]), axis=0)


            WL_10 = self.calc_woehlercurve(k=k_1, N0=ND50, S0=SD50, y_min=SD10, y_max=WC_data.loads_max*1.2)
            WL_10 = self.shift_woehlercurve_pf(WL50=WL_10, WoehlerCurve=WC_data, pa_goal=0.1, method=method)
            WL_10 = np.append(WL_10, np.array([[SD10, 1E9]]), axis=0)

            WL_90 = self.calc_woehlercurve(k=k_1, N0=ND50, S0=SD50, y_min=SD90, y_max=WC_data.loads_max*1.2)
            WL_90 = self.shift_woehlercurve_pf(WL50=WL_90, WoehlerCurve=WC_data, pa_goal=0.9, method=method)
            WL_90 = np.append(WL_90, np.array([[SD90, 1E9]]), axis=0)

        else:
            WL_50_new = self.calc_woehlercurve(k=slope, N0=ND50, S0=SD50, y_min=ylim_WL[0], y_max=SD50)
            WL_50 = np.append(WL_50, WL_50_new, axis=0)

            WL_10 = self.calc_woehlercurve(k=k_1, N0=ND50, S0=SD50, y_min=SD10, y_max=WC_data.loads_max*1.2)
            WL_10 = self.shift_woehlercurve_pf(WL50=WL_10, WoehlerCurve=WC_data, pa_goal=0.1, method=method)
            ND10 = WL_10[-1,-1]
            WL_10_new = self.calc_woehlercurve(k=slope, N0=ND10, S0=SD10, y_min=ylim_WL[0], y_max=SD10)
            WL_10 = np.append(WL_10, WL_10_new, axis=0)

            WL_90 = self.calc_woehlercurve(k=k_1, N0=ND50, S0=SD50,y_min=SD90, y_max=WC_data.loads_max*1.2)
            WL_90 = self.shift_woehlercurve_pf(WL50=WL_90, WoehlerCurve=WC_data, pa_goal=0.9, method=method)
            ND90 = WL_90[-1,-1]
            WL_90_new = self.calc_woehlercurve(k=slope, N0=ND90, S0=SD90, y_min=ylim_WL[0], y_max=SD90)
            WL_90 = np.append(WL_90, WL_90_new, axis=0)

        plt.plot(WL_50[:, 1], WL_50[:, 0], 'r', linewidth=2., label=u'WC, $P_A$=50%')
        plt.plot(WL_10[:, 1], WL_10[:, 0], 'r', linewidth=1.5, linestyle='--',
                 label=u'WC, $P_A$=10% u. 90%')
        plt.plot(WL_90[:, 1], WL_90[:, 0], 'r', linewidth=1.5, linestyle='--')

        # add_DL_labels(fig,data) #Anzahl der Durchläufer auf einem Niveau markieren
        plt.title(u'Woehler-Diagram')

        self.edit_WL_diagram(fig, amp, ld_typ, unit, xlim=xlim_WL, ylim=ylim_WL)

        text = '$k_1$ = '+str(np.round(k_1,decimals=2)) + '\n'
        text += '$k_2$ = '+str(np.round(slope,decimals=2)) + '\n'
        text += '$1/T_N$ = ' + str(np.round(TN,decimals=2)) + '\n'
        text += '$1/T_S^*$ = ' + str(np.round(TN**(1./k_1),decimals=2)) + '\n'
        text += '$S_{D,50}$ = ' + str(np.round(SD50,decimals=1)) + '\n'
        text += '$N_{D,50}$ = ' + '{:1.2e}'.format(ND50) + '\n'
        text += '$1/T_S$ = ' + str(np.round(TS,decimals=2))


        plt.text(0.01, 0.03, text,
                 verticalalignment='bottom',horizontalalignment='left',
                 transform=ax.transAxes, bbox={'facecolor':'grey', 'alpha':0.2, 'pad':10})

    def calc_woehlercurve_from_curve(self, WoehlerCurve, method):
        """ Basquin curve equation

        http://www.ux.uis.no/~hirpa/6KdB/ME/S-N%20diagram.pdf
        """
        if method == 'Mali':
            k = WoehlerCurve.Mali_5p_result['k_1']
            S0 = WoehlerCurve.Mali_5p_result['SD_50']
            N0 = WoehlerCurve.Mali_5p_result['ND_50']
        else:
            k = WoehlerCurve.k
            S0 = 1
            N0 = WoehlerCurve.N0
        y_min = WoehlerCurve.loads_max*1.2
        y_max = WoehlerCurve.loads_min*0.8

        self.wl_curve = self.calc_woehlercurve(k, N0, S0, y_min, y_max)

        return self.wl_curve
    
    def calc_woehlercurve(self, k, N0, S0, y_min, y_max):
        """ Basquin curve equation

        http://www.ux.uis.no/~hirpa/6KdB/ME/S-N%20diagram.pdf
        """
        y = np.linspace(y_max, y_min, num=100)
        x = N0*(y/S0)**(-k)

        return np.array([y, x]).transpose()

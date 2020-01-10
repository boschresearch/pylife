import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy import signal
from scipy import optimize as op


class psdSignal:
    '''Handles different routines for self signals
    
    Remark: We are using the pandas data frame schema. The index contains the
    discrete frequency step. Every single column one self. 
    
    Some functions of these class: 

    * psd_optimizer
    * ....   
    '''
    def __init__(self,df):
        
        self.df = df
    def _intMinlog(log_Hi,psdin,f,fsel,factor_rms_nods):
        # alles logarithmisieren
        logY = np.log10(psdin)
        logYsel =  np.interp(fsel,f,logY)# an den h-Stuetzstellen
        eps1 = ((sum(psdin)-sum(np.power(10,np.interp(f,fsel,log_Hi))))**2)/(sum(psdin)**2)
        eps2 =  np.dot(logYsel-log_Hi,logYsel-log_Hi)/np.dot(logYsel,logYsel)
        return factor_rms_nods*eps1+(1-factor_rms_nods)*eps2
    
    
    def psd_smoother(self,fsel,factor_rms_nodes = 0.5):
        ''' Smooth your PSD using nodes and a penalty factor weighting the errors
        for the RMS and for the node PSD values
        
        Parameters:
        ----------
        
        self: DataFrame
            unsmoothed PSD        
        fsel: list or np.array
           nodes
        factor_rms_nodes: float (0 <= factor_rms_nods <= 1)
            penalty error weighting the errors:
                
            * 0: only error of node PSD values is considered
            * 1: only error of the RMS is considered
            
        Returns:
        --------
        DataFrame        
        '''
    	# InputVariablen
    	# 
        f  = np.log10(self.index.values)
        fsel = np.log10(np.unique(np.append(min(f),fsel)))#,max(f)]))
        opt_df = pd.DataFrame()
        for colact in self.columns:
            log_Hi0 = np.interp(fsel,f,np.log10(self[colact]))
            log_Hi = op.fmin(psdSignal._intMinlog,log_Hi0,args=(self[colact],f,fsel,factor_rms_nodes),disp = 0)
            log_H = np.interp(f,fsel,log_Hi)
            opt_df[colact] = np.power(10,log_H)
        opt_df.index = self.index            
        return opt_df
    
    
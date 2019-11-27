import pandas as pd
import numpy as np
from scipy import optimize as op
def psd_optimizer(psd,fsel,factor_rms_knots = 0.5):
	# InputVariablen
	# 
    f  = psd.index.values
    fsel = np.unique(np.append(min(f),fsel))#,max(f)]))
    log_Hi0 = np.interp(fsel,f,np.log10(psd))
    log_Hi = op.fmin(_intMinlog,log_Hi0,args=(psd,f,fsel,factor_rms_knots))
    log_H = np.interp(f,fsel,log_Hi)
    psd_opt = np.power(10,log_H)
    return pd.Series(data = psd_opt,index = psd.index)
def _intMinlog(log_Hi,PSDin,f,fsel,factor_rms_knots):
    # alles logarithmisieren
    logY = np.log10(PSDin)
    logYsel =  np.interp(fsel,f,logY)# an den h-Stuetzstellen
    eps1 = ((sum(PSDin)-sum(np.power(10,np.interp(f,fsel,log_Hi))))**2)/(sum(PSDin)**2)
    eps2 =  np.dot(logYsel-log_Hi,logYsel-log_Hi)/np.dot(logYsel,logYsel)
    return factor_rms_knots*eps1+(1-factor_rms_knots)*eps2


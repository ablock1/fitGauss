import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def fitSlope(t,sigSq, **kwargs):
    tStart = t[0]
    tEnd = t[-1]
    if kwargs is not None:
        for key, value in kwargs.items():
            if key == 'tStart':
                tStart = value
            elif key == 'tEnd':
                tEnd = value
    t_ind_bool = np.all([t>=tStart,t<=tEnd],axis=0)
    p0 = np.polyfit(t[t_ind_bool], sigSq[t_ind_bool], 1)
    return p0[0], p0[1], 0.5*p0[0], t_ind_bool
    
def gaus(x,a,x0,sigma):
    return a*np.exp(-0.5*(x-x0)**2/(sigma**2))

def gausCutTail(x,a,x0,sigma, N):
    '''make a Gaussian, but set to zero outside the region x > N*sigma '''
    y = a*np.exp(-0.5*(x-x0)**2/(sigma**2))
    mask1 = (x-x0>=-sigma*N)
    mask2 = (x-x0<=sigma*N)
    return y*mask1*mask2

def fitGaussLinesTo2Ddataset(x,YY,typee,**kwargs):
    '''assume the dataset is a rectangle with rows that should be fit (hence x=x-axis) '''
    trans = False
    plotAll = False
    if kwargs is not None:
        for key, value in kwargs.items():
            if key == 'transpose' and value == True:
                trans = True
                YY = YY.transpose()
            if key == 'plotAll' and value == True:
                plotAll = True
    if len(x) == np.shape(YY)[0]:
        numOfRows = np.shape(YY)[1]
        resids = np.zeros( np.shape(YY) )
        sig, sigErr = ( np.zeros( numOfRows ), np.zeros( numOfRows ) )
        for j in range(numOfRows):
            if typee=='noOffs':
                try:
                    _, _, _,  resids[:,j], sig[j], sigErr[j] = fitOneGaussianNoOffset(x,YY[:,j],plotAll)
                except ValueError:
                    print('ValueError in GaussFit (noOffs): set all zeros.')
                    resids[:,j], sig[j], sigErr[j] = ( np.zeros(np.shape(x)), 0, 0 )
                except:
                    print('GaussFit: Something else went wrong')                        
            elif typee == 'wOffs':
                try:
                    _, _, _,  resids[:,j], sig[j], sigErr[j] = fitOneGaussianWithOffset(x,YY[:,j],plotAll)
                except ValueError:
                    print('ValueError in GaussFit(wOffs): set all zeros.')
                    resids[:,j], sig[j], sigErr[j] = (np.zeros(np.shape(x)), 0, 0)
                except:
                    print('GaussFit: Something else went wrong') 
            else:
                print('error: typee not recognized')
            if trans:
                resids = resids.transpose()
        sigSq = sig**2
        sigSqErr = 2*sig*sigErr
    else:
        print('GaussFitError: 2D dataset dimension 1 not same length as x')
    return sig, sigErr, resids, sigSq, sigSqErr

def fitOneGaussianNoOffset(x,y,showPlot):
    x,y = (np.float64(x), np.float64(y))
    maxx = np.amax(y)
    minn = np.amin(y)
    if max(np.abs(maxx),np.abs(minn)) == 0:
        return 0,0,'all zeros',np.zeros(len(x)), 0, 0
        print('all zeros found in GaussFit')
    else:
        if np.abs(minn)>np.abs(maxx):
            amp0 = minn
        else:
            amp0 = maxx
        x00 = np.sum(x*y)/np.sum(y)
        if x00<x[0] or x00>x[-1]:
            print(f'x00 estimation {x00:.2f} outside x vector ?')
            x00 = 0.5*(x[0] + x[-1])
        sigma0 = np.sqrt(sum((y)/sum(y)*(x-x00)**2))
        #print(f'amp0 = {amp0:.2e}, x00 = {x00:.2e}, sigma0 = {sigma0:.2e}')
        bounds0 = ([-np.inf, np.amin(x), 0],[np.inf, np.amax(x), np.abs(x[-1]-x[0])]) #11/2019 changed x0 bounds from min([x[-1],x[0]]) and max([x[-1],x[0]]), which made no sense.
        popt,pcov = curve_fit(gaus,x,y,p0=[amp0,x00,sigma0], bounds = bounds0)
        perr = np.sqrt(np.diag(pcov))
        sig = popt[2]
        FW = 2*np.sqrt(2*np.log(2))*popt[2]
        sigErr = perr[2]
        FWerr = 2*np.sqrt(2*np.log(2))*perr[2]
        FWtext = 'FW = {:.2f} $\pm$ {:.2f}'.format(FW,FWerr)
        resids = y - gaus(x, *popt)
        if showPlot:
            f2 = plt.figure()
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax1.plot(x,y,'bx',label='data')
            ax1.plot(x,gaus(x,*popt),'r-',label='fit:' +FWtext)
            ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax2.plot(x,resids,'k.',label='residuals')
            ax1.set_xlabel('μm')
            ax1.set_ylabel('Amplitude (a.u.)')
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Residuals (a.u.)')
            ax1.set_title('Gaussian Fit')
            ax1.legend()
            f2.tight_layout()
        return FW, FWerr, FWtext, resids, sig, sigErr

def exampleGaussFit():    
    a1 = 10
    a2 = 1
    FW1 = 1
    FW2 = 3
    beta = 4*np.log(2)
    x1 = np.arange(-10,10.001,0.2)
    y1 = a1*np.exp(-beta*x1**2/FW1**2)+a2*np.exp(-beta*x1**2/FW2**2)
    FW,FW_err, FW_text, resids = fitOneGaussianNoOffset(x1,y1,1)

def gausWoffset(x,a,x0,sigma,offs):
    return a*np.exp(-0.5*(x-x0)**2/(sigma**2))+offs

def fitOneGaussianWithOffset(x,y,showPlot):
    x,y = (np.float64(x), np.float64(y))
    offs0 = y[-1]
    maxx = np.amax(y-offs0)
    minn = np.amin(y-offs0)
    if max(np.abs(maxx),np.abs(minn)) == 0:
        return 0,0,'all zeros',np.zeros(len(x))
        print('all zeros found in GaussFit')
    else:
        if np.abs(minn)>np.abs(maxx):
            amp0 = minn
        else:
            amp0 = maxx
        x00 = np.sum(x*y)/np.sum(y) 
        sigma0 = np.sqrt(np.abs(np.sum((y-offs0)/np.sum(y-offs0)*(x-x00)**2))) 
        bounds0 = ([-np.inf, min(x[-1],x[0]), 0,-np.inf],[np.inf, max(x[-1],x[0]), 10*np.abs(x[-1]-x[0]),np.inf])
#        print(f'amp0 = {amp0:.2e}, x00 = {x00:.2e}, sigma0 = {sigma0:.2e}, offs0 = {offs0:.2e}')
#        print(bounds0)
        popt,pcov = curve_fit(gausWoffset,x,y,p0=[amp0,x00,sigma0,offs0], bounds = bounds0 )
        perr = np.sqrt(np.diag(pcov))
        sig = popt[2]
        FW = 2*np.sqrt(2*np.log(2))*popt[2]
        sigErr = perr[2]
        FWerr = 2*np.sqrt(2*np.log(2))*perr[2]
        FWtext = 'FW = {:.2f} $\pm$ {:.2f}'.format(FW,FWerr)
        resids = y - gausWoffset(x, *popt)
        if showPlot:
            f2 = plt.figure()
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax1.plot(x,y,'bx',label='data')
            ax1.plot(x,gausWoffset(x,*popt),'r-',label='fit:' +FWtext)
            ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax2.plot(x,resids,'k.',label='residuals')
            ax1.set_xlabel('μm')
            ax1.set_ylabel('Amplitude (a.u.)')
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Residuals (a.u.)')
            ax1.set_title('Gaussian Fit')
            ax1.legend()
            f2.tight_layout()
        return FW, FWerr, FWtext, resids, sig, sigErr

def exampleGaussFitWoffs():    
    a1 = 10
    a2 = 1
    offs = 3
    FW1 = 1
    FW2 = 3
    beta = 4*np.log(2)
    x1 = np.arange(-10,10.001,0.2)
    y1 = a1*np.exp(-beta*x1**2/FW1**2)+a2*np.exp(-beta*x1**2/FW2**2)+offs
    FW,FW_err, FW_text, resids = fitOneGaussianWithOffset(x1,y1,1)

def gausRad(x,a,sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))

def fitRadialGaussNoOffset(r,y,showPlot):
    maxx = np.amax(y)
    minn = np.amin(y)
    if max(np.abs(maxx),np.abs(minn)) == 0:
        return 0,0,'all zeros',np.zeros(len(r))
        print('all zeros found in GaussFit')
    else:
        if np.abs(minn)>np.abs(maxx):
            amp0 = minn
        else:
            amp0 = maxx
        sigma0 = np.sqrt(np.sum((y)/np.sum(y)*(r)**2))
        #print(f'amp0 = {amp0:.2e}, x00 = {x00:.2e}, sigma0 = {sigma0:.2e}')
        popt,pcov = curve_fit(gausRad,r,y,p0=[amp0,sigma0], bounds = ([-np.inf, 0],[np.inf, np.abs(r[-1]-r[0])]) )#p0=[amp0,sigma0])
        perr = np.sqrt(np.diag(pcov))
        FW = 2*np.sqrt(2*np.log(2))*popt[1]
        FW_err = 2*np.sqrt(2*np.log(2))*perr[1]
        FW_text = 'FW = {:.2f} $\pm$ {:.2f}'.format(FW,FW_err)
        resids = y - gausRad(r, *popt)
        if showPlot:
            f2 = plt.figure()
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax1.plot(r,y,'bx',label='data')
            ax1.plot(r,gausRad(r,*popt),'r-',label='fit:' +FW_text)
            ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax2.plot(r,resids,'k.',label='residuals')
            ax1.set_xlabel('μm')
            ax1.set_ylabel('Amplitude (a.u.)')
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Residuals (a.u.)')
            ax1.set_title('Gaussian Fit')
            ax1.legend()
            f2.tight_layout()
        return FW,FW_err, FW_text, resids
    
def exampleRadGaussNoOffsFit():
    a1 = 10
    a2 = 1
    FW1 = 1
    FW2 = 3
    beta = 4*np.log(2)
    r1 = np.arange(0.2,10.001,0.2)
    y1 = a1*np.exp(-beta*r1**2/FW1**2)+a2*np.exp(-beta*r1**2/FW2**2)
    FW,FW_err, FW_text, resids = fitRadialGaussNoOffset(r1,y1,1)

def gausRadWOffs(x,a,sigma, offs):
    return offs + a*np.exp(-(x)**2/(2*sigma**2))

def fitRadialGaussWithOffset(r,y,showPlot):
    offs0 = y[-1]
    maxx = np.amax(y-offs0)
    minn = np.amin(y-offs0)
    if max(np.abs(maxx),np.abs(minn)) == 0:
        return 0,0,'all zeros',np.zeros(len(r))
        print('all zeros found in GaussFit')
    else:
        if np.abs(minn)>np.abs(maxx):
            amp0 = minn
        else:
            amp0 = maxx
        sigma0 = np.sqrt(np.abs(np.sum((y-offs0)/np.sum(y-offs0)*(r)**2)))
        #print(f'amp0 = {amp0:.2e}, sigma0 = {sigma0:.2e}, offs0 = {offs0:.2e}')
        popt,pcov = curve_fit(gausRadWOffs,r,y,p0=[amp0, sigma0, offs0], bounds = ([-np.inf, 0, -np.inf],[np.inf, np.abs(r[-1]-r[0]), np.inf]) )#p0=[amp0,sigma0, offs0])
        perr = np.sqrt(np.diag(pcov))
        FW = 2*np.sqrt(2*np.log(2))*popt[1]
        FW_err = 2*np.sqrt(2*np.log(2))*perr[1]
        FW_text = f'FW = {FW:.2f} $\pm$ {FW_err:.2f}'
        resids = y - gausRadWOffs(r, *popt)
        if showPlot:
            plt.ion()
            f2 = plt.figure()
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax1.plot(r,y,'bx',label='data')
            ax1.plot(r,gausRadWOffs(r,*popt),'r-',label='fit:' +FW_text)
            ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            ax2.plot(r,resids,'k.',label='residuals')
            ax1.set_xlabel('μm')
            ax1.set_ylabel('Amplitude (a.u.)')
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Residuals (a.u.)')
            ax1.set_title('Gaussian Fit')
            ax1.legend()
            f2.tight_layout()
            plt.show(block=False)
        return FW,FW_err, FW_text, resids

def exampleRadGaussWOffsFit():
    offs0 = 293
    a1 = 10
    a2 = 1
    FW1 = 1
    FW2 = 3
    beta = 4*np.log(2)
    r1 = np.arange(0.2,10.001,0.2)
    y1 = a1*np.exp(-beta*r1**2/FW1**2)+a2*np.exp(-beta*r1**2/FW2**2) + offs0
    FW,FW_err, FW_text, resids = fitRadialGaussWithOffset(r1,y1,1)
#%%
#dx = 0.4
#x = np.arange(-10,10.001,dx) # um
#plt.figure()
#plt.plot(x, (gausCutTail(x,a=1,x0=0,sigma=1.1, N=2.0)),'x--')
#exampleGaussFit()
#exampleRadGaussWOffsFit()
#exampleRadGaussNoOffsFit()
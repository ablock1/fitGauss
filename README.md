# fitGauss
fit single Gauss to data

use like this:

FW, FWerr, FWtext, resids, sig, sigErr = fitOneGaussianNoOffset(x,y,showPlot)

Input: 
x: x-data (1D array)
y: y-data (1D array)

Output:
FW: FWHM width (not sigma!) (float)
FWerr: fit width error (xx% confidence) (float)
resids: residual data (1D array)
sig: "sigma" width = FWHM / 2.355 
sigErr: fit sigma error

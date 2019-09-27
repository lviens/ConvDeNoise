import numpy as np
import scipy
import scipy.signal
from scipy.signal import butter, filtfilt

#%%
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Two-pass Butterworth filter
    Inputs: 
        - data: Data to filter (in time domain)
        - lowcut: Lowest frequency of the filter (in Hz)
        - highcut: Highest frequency of the filter (in Hz)
        - fs: Sampling rate (in Hz)
        - Order of the filter (4)
    Output:
        y: Filtered data
    
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def ncf_denoise(img_to_denoise, mdate, ntau, nsv, nsv_to_rm, use_wiener):
    """
    SVDWF method from Moreau et al (2017)
    Inputs:
        - img_to_denoise: the list of NCF. It should be an MxN matrix where M represents the total number of NCF and N the
            length of each NCF
        - mdate: the size of the Wiener filter in the first dimension (K = 5)
        - ntau: the size of the Wiener filter in the second dimension (L = 5)
        - nsv: the number of singular values to keep in the SVD filter (25)
    Outputs:
        - denoised_img: the denoised list of NCF
    """
    if nsv > min(np.shape(img_to_denoise)):
        nsv = min(np.shape(img_to_denoise))

    U, s, V = np.linalg.svd(img_to_denoise)
    Xwiener = np.zeros((np.shape(img_to_denoise)))

    for kk in np.arange(nsv_to_rm, nsv):
        SV = np.zeros((img_to_denoise.shape))
        SV[kk, kk] = s[kk]
        X = U @ SV @ V  # equivalently, U.dot(SV.dot(V))
        Xwiener = scipy.signal.wiener(X, [mdate, ntau]) + Xwiener
    if use_wiener is True:
        denoised_img = scipy.signal.wiener(Xwiener, [mdate, ntau])
    else:
        denoised_img = Xwiener

    return denoised_img



def stretching_current(ref, cur, dvmin, dvmax, nbtrial, window,t_vec):
    """
    Function to perform the stretching of the waveforms: 
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.
    INPUTS:
        - ref = Reference waveform (np.ndarray, size N)
        - cur = Current waveform (np.ndarray, size N)
        - dvmin = minimum bound for the velocity variation; example: dvmin=-0.03 for -3% of relative velocity change ('float')
        - dvmax = maximum bound for the velocity variation; example: dvmax=0.03 for 3% of relative velocity change ('float')
        - nbtrial = number of stretching coefficient between dvmin and dvmax, no need to be higher than 50  ('float')
        - window = vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
        - t_vec = time vector, common to both ref and cur (np.ndarray, size N)

    OUTPUTS:
        - dv = Relative velocity change dv/v (in %)
        - cc = correlation coefficient between the reference waveform and the best stretched/compressed current waveform
        - Eps = Vector of Epsilon values (Epsilon =-dt/t = dv/v)
    """
    Eps = np.asmatrix(np.linspace(dvmin, dvmax, nbtrial))
    L = 1 + Eps
    tt = np.matrix.transpose(np.asmatrix(t_vec))
    tau = tt.dot(L)  # stretched/compressed time axis
    C = np.zeros((1, np.shape(Eps)[1]))

    for j in np.arange(np.shape(Eps)[1]):
        s = np.interp(x=np.ravel(tt), xp=np.ravel(tau[:, j]), fp=cur)
        waveform_ref = ref[window]
        waveform_cur = s[window]
        C[0, j] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]

    imax = np.nanargmax(C)
        
    if imax >= np.shape(Eps)[1]-1:
        imax = imax - 1
    if imax <= 2:
        imax = imax + 1
    dtfiner = np.linspace(Eps[0, imax-1], Eps[0,imax+1], 500)
    func = scipy.interpolate.interp1d(np.ravel(Eps[0,np.arange(imax-2, imax+2)]), np.ravel(C[0,np.arange(imax-2, imax+2)]), kind='cubic')
    CCfiner = func(dtfiner)
    cc = np.max(CCfiner) # Find maximum correlation coefficient of the refined  analysis
    dv = 100 * dtfiner[np.argmax(CCfiner)]  # Final dv/v measurement (x100 to get the dv/v in %)
    

    return dv, cc, Eps

def naff1d_gauss(z, alpha=140, Offset=False):
    """Naff 1d with gauss-window weights; approximation"""
    N = z.shape[0]
    t = np.arange(N)
    weights = np.exp(-alpha * (t / N - 0.5)**2)
    abs_fft = np.abs(np.fft.fft(weights * z))
    
    # correct a potential constant offset of the input signal
    # it is much better to perform this after the FFT (else O(1/N) error)
    if Offset:
        c0 = np.sum(z)
        J = 30    # number of corrections until 1e-16 error
        if J > N//2: # avoid index error for small 'N'
            J = N//2
        corr = np.abs(c0) * np.exp(-np.pi**2 / alpha * np.arange(J)**2)
        abs_fft[:J] -= corr
        # correction for the 'negative frequency' section (J-1 terms)
        abs_fft[-J+1:] -= corr[:0:-1]   
    
    # find the global maximum of the FFT
    ind = np.argmax(abs_fft)
    if ind == (N-1):   # avoid index error at 'abs_fft[ind + 1]'
        ind -= 1
    elif abs_fft[ind - 1] > abs_fft[ind + 1]:   # 'ind' closer to true peak
        ind -= 1
    R = abs_fft[ind] / abs_fft[ind + 1]
    nu = ind / N + 1 / (2*N) - alpha / (2 * np.pi**2 * N) * np.log(R)
    return nu
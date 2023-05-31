import numpy as np

from scipy.stats import f
from scipy.stats import skew

import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.table import Table

import warnings
import os


def chisq(data, model):
    return np.sum(((data - model) / data)**2)


def vonNeumann(x):
    return (np.nansum(np.diff(x)**2) / (x.size - 1)) / (np.nanstd(x)**2)


def conStat(values, threshold=2., n_con=3):
    '''
    See Wozniak 2000.
    Return Wozniak index = (number of consecutive series / (N - 2))
    See Shin et al. 2009 as well.

    values : list of magnitudes.

    threshold : limit of consecutive points.

    n_con : miminum number of consecutive points to generate a consecutive series.
    '''

    median_value = np.median(values)
    std_value = np.std(values)
    n_value = len(values)

    # remember that this is magnitude based values.
    bright_index = np.where(values <= median_value - threshold * std_value)[0]
    faint_index = np.where(values >= median_value + threshold * std_value)[0]
    if n_value <= 2:
        return 0.

    def find_con_series(indices):
        n_con = 0
        # pre_i = -1
        for i in range(1, len(indices) - 1):
            if indices[i] == (indices[i - 1] + indices[i + 1]) / 2:
                # if i != pre_i + 1:
                n_con += 1
                # pre_i = i

        # count all the lowest points of the consecutive indices
        for i in range(2, len(indices) - 1):
            if (indices[i] == (indices[i - 1] + indices[i + 1]) / 2) and (indices[i - 1] != (indices[i - 2] + indices[i]) / 2):
                n_con += 1

        # count all the highest points of the consecutive indices
        for i in range(1, len(indices) - 2):
            if (indices[i] == (indices[i - 1] + indices[i + 1]) / 2) and (indices[i + 1] != (indices[i + 2] + indices[i]) / 2):
                n_con += 1

        return n_con

    return (find_con_series(bright_index) + find_con_series(faint_index)) / float(n_value - 2)


def process_LC(lc_data, fltRange=5.0, detrend=False, detrend_deg=3):
    lc_mjd = lc_data['mjd']
    lc_mag = lc_data['mag']
    lc_err = lc_data['magerr']

    Tspan100 = lc_mjd.max() - lc_mjd.min()
    Tspan95 = np.percentile(lc_mjd, 100 - fltRange) - np.percentile(lc_mjd, fltRange)

    nmag = len(lc_mag)

    # magmn = np.mean(lc_mag)
    errmn = np.mean(lc_err)

    brt10per = np.percentile(lc_mag, fltRange)
    fnt10per = np.percentile(lc_mag, 100 - fltRange)
    brt10data = lc_data[lc_mag <= brt10per]
    fnt10data = lc_data[lc_mag >= fnt10per]

    medmagfnt10 = np.median(fnt10data['mag'])
    mederrfnt10 = np.median(fnt10data['magerr'])

    medmagbrt10 = np.median(brt10data['mag'])
    mederrbrt10 = np.median(brt10data['magerr'])

    brtcutoff = medmagbrt10 - (2 * mederrbrt10)
    fntcutoff = medmagfnt10 + (2 * mederrfnt10)

    # filter_data = lc_data[(lc_mag >= brtcutoff) & (lc_mjd >= 0.0)]
    # flc_data = filter_data[filter_data['mag'] <= fntcutoff]

    filter_index = (lc_mag >= brtcutoff) & (lc_mag <= fntcutoff) & (lc_mjd >= 0.0)
    lc_data.add_column(filter_index, name='QualFlag')

    flc_data = lc_data[filter_index]

    # flc_data = flc_data[flc_data['magerr'] > 0.05]

    flc_mag = flc_data['mag']
    flc_mjd = flc_data['mjd']
    flc_err = flc_data['magerr']
    flc_nmag = len(flc_mag)

    if flc_nmag >= 10:
        p_lin, residuals_lin, rank_lin, singular_values_lin, rcond_lin = np.polyfit(flc_mjd - flc_mjd.min(), flc_mag, 1, rcond=None, full=True, w=1.0 / flc_err, cov=False)  # linear fit to LC
        p_quad, residuals_quad, rank_quad, singular_values_quad, rcond_quad = np.polyfit(flc_mjd - flc_mjd.min(), flc_mag, 2, rcond=None, full=True, w=1.0 / flc_err, cov=False)  # Quadratic fit to LC
        residuals_lin *= (1. / (flc_nmag - 1))
        residuals_quad *= (1. / (flc_nmag - 1))
    else:
        p_lin = np.array([0, 0])
        residuals_lin = np.array([99999.9999])
        p_quad = np.array([0, 0, 0])
        residuals_quad = np.array([99999.9999])

    mean_error = np.nanmean(flc_err)
    lc_std = np.nanstd(flc_mag)
    var_stat = lc_std / mean_error

    con = conStat(flc_mag)

    fmagmed = np.median(flc_mag)
    fmagmn = np.mean(flc_mag)
    fmagmax = np.max(flc_mag)
    fmagmin = np.min(flc_mag)

    # ferrmed = np.median(flc_err)
    ferrmn = np.mean(flc_err)
    # fmag_stdev = np.std(flc_mag)

    rejects = nmag - flc_nmag

    mag_above = np.mean(flc_mag) - (2 * ferrmn)
    mag_below = np.mean(flc_mag) + (2 * ferrmn)

    nabove = np.where(flc_mag <= mag_above)[0].size
    nbelow = np.where(flc_mag >= mag_below)[0].size

    Mt = (fmagmax - fmagmed) / (fmagmax - fmagmin)

    minpercent = np.percentile(flc_mag, 5)
    maxpercent = np.percentile(flc_mag, 95)

    a95 = maxpercent - minpercent

    lc_skew = skew(flc_mag)
    lc_vonNeumann = vonNeumann(flc_mag)

    summation_eqn1 = np.sum(((flc_mag - fmagmn)**2) / (flc_err**2))
    if flc_nmag <= 1:
        Chi2 = np.inf
    else:
        Chi2 = (1. / (flc_nmag - 1)) * summation_eqn1

    properties = {'Tspan100': Tspan100, 'Tspan95': Tspan95, 'a95': a95, 'Mt': Mt, 'lc_skew': lc_skew,
                  'Chi2': Chi2, 'brtcutoff': brtcutoff, 'brt10per': brt10per, 'fnt10per': fnt10per,
                  'fntcutoff': fntcutoff, 'errmn': errmn, 'ferrmn': ferrmn, 'ngood': flc_nmag,
                  'nrejects': rejects, 'nabove': nabove, 'nbelow': nbelow, 'VarStat': var_stat, 'vonNeumann': lc_vonNeumann,
                  'Con': con, 'm': p_lin[0], 'b_lin': p_lin[1], 'chi2_lin': residuals_lin[0],
                  'a': p_quad[0], 'b_quad': p_quad[1], 'c': p_quad[2], 'chi2_quad': residuals_quad[0]}

    # flc_data = [flc_mjd, flc_mag, flc_err]
    # return Table(flc_data), properties
    if detrend:
        trend_fit = np.polyfit(lc_data['mjd'], lc_data['mag'], deg=detrend_deg, w=1 / lc_data['magerr'])
        rescaled_mag = (lc_data['mag'] / np.polyval(trend_fit, lc_data['mjd'])) * np.nanmean(lc_data['mag'])
        lc_data['mag'] = rescaled_mag

    return lc_data, properties


class MultiTermFit:
    """Multi-term Fourier fit to a light curve
    Parameters
    ----------
    omega : float
        angular frequency of the fundamental mode
    n_terms : int
        the number of Fourier modes to use in the fit
    """

    def __init__(self, omega, n_terms):
        self.omega = omega
        self.n_terms = n_terms

    def _make_X(self, t):
        t = np.asarray(t)
        k = np.arange(1, self.n_terms + 1)
        X = np.hstack([np.ones(t[:, None].shape),
                       np.sin(k * self.omega * t[:, None]),
                       np.cos(k * self.omega * t[:, None])])
        return X

    def fit(self, t, y, dy):
        """Fit multiple Fourier terms to the data
        Parameters
        ----------
        t: array_like
            observed times
        y: array_like
            observed fluxes or magnitudes
        dy: array_like
            observed errors on y
        Returns
        -------
        self :
            The MultiTermFit object is  returned
        """
        t = np.asarray(t)
        y = np.asarray(y)
        dy = np.asarray(dy)

        self.y_ = y
        self.dy_ = dy
        self.t_ = t

        self.C_ = np.diag(dy * dy)

        X_scaled = self._make_X(t) / dy[:, None]
        y_scaled = y / dy

        self.w_ = np.linalg.solve(np.dot(X_scaled.T, X_scaled),
                                  np.dot(X_scaled.T, y_scaled))
        self.cov_ = np.linalg.inv(np.dot(X_scaled.T, X_scaled))
        return self

    def predict(self, Nphase, return_phased_times=False, adjust_offset=True):
        """Compute the phased fit, and optionally return phased times
        Parameters
        ----------
        Nphase : int
            Number of terms to use in the phased fit
        return_phased_times : bool
            If True, then return a phased version of the input times
        adjust_offset : bool
            If true, then shift results so that the minimum value is at phase 0
        Returns
        -------
        phase, y_fit : ndarrays
            The phase and y value of the best-fit light curve
        phased_times : ndarray
            The phased version of the training times.  Returned if
            return_phased_times is set to  True.
        """
        phase_fit = np.linspace(0, 1, Nphase + 1)[:-1]

        X_fit = self._make_X(2 * np.pi * phase_fit / self.omega)
        y_fit = np.dot(X_fit, self.w_)
        i_offset = np.argmin(y_fit)

        if adjust_offset:
            y_fit = np.concatenate([y_fit[i_offset:], y_fit[:i_offset]])

        if return_phased_times:
            if adjust_offset:
                offset = phase_fit[i_offset]
            else:
                offset = 0
            phased_times = (self.t_ * self.omega * 0.5 / np.pi - offset) % 1

            return phase_fit, y_fit, phased_times

        else:
            return phase_fit, y_fit

    def modelfit(self, time):
        y_t = self.w_[0]
        for ii in range(self.n_terms):
            coefs = self.w_[(2 * ii + 1):(2 * ii + 3)]
            y_t += coefs[0] * np.sin((ii + 1) * self.omega * time) + coefs[1] * np.cos((ii + 1) * self.omega * time)
        return y_t


def AFD(data, period, alpha=0.99, Nmax=6):
    """Automatic Fourier Decomposition
    """
    omega = (2.0 * np.pi) / period
    mjd, mag, err = data

    add_another_term = True
    # alpha = 0.99
    Nterms1 = 1
    while add_another_term:
        fstats = np.zeros(2)
        test_values = np.zeros(2)

        if Nterms1 >= Nmax:
            add_another_term = False
            best_Nterms = Nmax

        for Nterms2 in np.arange(Nterms1 + 1, Nterms1 + 3):
            n_fit_params1 = (Nterms1 * 2) + 1
            n_fit_params2 = (Nterms2 * 2) + 1

            mtf1 = MultiTermFit(omega, Nterms1)
            mtf1.fit(mjd, mag, err)
            phase_fit1, y_fit1, phased_t1 = mtf1.predict(1000, return_phased_times=True, adjust_offset=True)
            inter_y_fit1 = np.interp(phased_t1, phase_fit1, y_fit1)

            resid1 = mag - inter_y_fit1
            ChiSq1 = np.sum((resid1 / err)**2)
            reduced_ChiSq1 = ChiSq1 / (resid1.size - n_fit_params1)

            mtf2 = MultiTermFit(omega, Nterms2)
            mtf2.fit(mjd, mag, err)
            phase_fit2, y_fit2, phased_t2 = mtf2.predict(1000, return_phased_times=True, adjust_offset=True)
            inter_y_fit2 = np.interp(phased_t2, phase_fit2, y_fit2)

            resid2 = mag - inter_y_fit2
            ChiSq2 = np.sum((resid2 / err)**2)
            reduced_ChiSq2 = ChiSq2 / (resid2.size - n_fit_params2)

            b = resid2.size - n_fit_params2
            a = n_fit_params2 - n_fit_params1

            # fstat = (ChiSq1 / ChiSq2 - 1) * (b / a)
            fstats[Nterms2 - Nterms1 - 1] = (reduced_ChiSq1 / reduced_ChiSq2 - 1) * (b / a)

            df1 = resid1.size - n_fit_params1  # a
            df2 = resid2.size - n_fit_params2  # b
            test_values[Nterms2 - Nterms1 - 1] = f.ppf(alpha, dfn=df1, dfd=df2)

        if (fstats[0] <= test_values[0]) & (fstats[1] <= test_values[1]):
            add_another_term = False
            best_Nterms = Nterms1
        elif (fstats[0] <= test_values[0]) & ((fstats[0] > test_values[0])):
            Nterms1 += 2
        else:
            Nterms1 += 1

    best_mtf = MultiTermFit(omega, best_Nterms)
    best_mtf.fit(mjd, mag, err)
    best_phase_fit, best_y_fit, best_phased_t = best_mtf.predict(1000, return_phased_times=True, adjust_offset=True)
    best_inter_y_fit = np.interp(best_phased_t, best_phase_fit, best_y_fit)

    best_resid = mag - best_inter_y_fit
    best_ChiSq = np.sum((best_resid / err)**2)
    best_reduced_ChiSq = best_ChiSq / (best_resid.size - n_fit_params2)

    return best_Nterms, best_phase_fit, best_y_fit, best_phased_t, best_resid, best_reduced_ChiSq, best_mtf

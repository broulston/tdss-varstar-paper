# The source of this file lives locally at:
# /Users/benjaminroulston/Dropbox/Research/Projects/Variable_Stars/WORKING_DIRECTORY/PLOTS/Sample_plots/sample_plots.py
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import paths

prop = Table.read(paths.data / 'TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits')

# ra = prop['ra_GaiaEDR3'] * u.deg
# dec = prop['decGaiaEDR3'] * u.deg

# c = SkyCoord(ra=ra, dec=dec, frame='icrs')

# ra_rad = c.ra.wrap_at(180 * u.deg).radian
# dec_rad = c.dec.radian 

# ***********************************
# ***********************************
# ***********************************

# plt.figure(figsize=(16, 8))
# plt.subplot(111, projection="aitoff")
# plt.grid(True)
# plt.scatter(ra_rad, dec_rad, c='k', s=1, alpha=0.3)
# ax = plt.gca()
# ax.set_xticklabels(['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h', '12h'])
# plt.show()

# ***********************************
# ***********************************
# ***********************************

specType = prop['PyHammerSpecType']
specType = np.array([ii.strip().decode() for ii in specType.data])

maintype = np.array([ii[0] for ii in specType])
letterSpt = np.array(['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'd', 'D'])
letterSpt_forlabel = np.array(['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'C', 'DA'])
maintypeNum = np.array([np.where(letterSpt == ii)[0][0] for ii in maintype])

# np.where(maintypeNum == 7)

singleSpecType = np.array([ii for ii in specType if "+" not in ii])
SB2eSpecType = np.array([ii for ii in specType if "+" in ii])

unique_single, label, counts = np.unique(singleSpecType, return_inverse=True, return_counts=True)

u = prop['umag']
g = prop['gmag']
r = prop['rmag']
i = prop['imag']
z = prop['zmag']

g_gaia = prop['phot_g_mean_mag']
bp = prop['phot_bp_mean_mag']
rp = prop['phot_rp_mean_mag']
bp_rp = prop['bp_rp']
rest = prop['rpgeo']
M_G = g_gaia + 5 - 5 * np.log10(rest)

umg = u - g
gmr = g - r
rmi = r - i
imz = i - z

# ***********************************
# ***********************************
# ***********************************

plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

cmap = plt.get_cmap('nipy_spectral', letterSpt.size)
cmap_r = plt.get_cmap('nipy_spectral_r', letterSpt.size)

plt.figure(figsize=(9, 8))
ax = plt.gca()
sc = plt.scatter(umg, gmr, c=maintypeNum, s=2.5, alpha=0.3, cmap=cmap, vmin=-0.5, vmax=9.5)
sc1 = plt.scatter(umg, gmr, c=maintypeNum, s=0.0, alpha=1.0, cmap=cmap_r, vmin=-0.5, vmax=9.5)

divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(sc1, cax=cax1, ticks=np.arange(letterSpt.size))
cbar1.ax.get_yaxis().labelpad = 10
cbar1.ax.set_ylabel('SpecType', rotation=270)
cbar1.ax.set_yticklabels(np.flip(letterSpt_forlabel))

ax.set_xlabel("$u - g$")
ax.set_ylabel("$g - r$")
ax.set_xlim([-1.0, 5.0])
ax.set_ylim([-0.5, 2.0])

# ax.invert_yaxis()

ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

plt.tight_layout()
plt.savefig(paths.figures / "gmr_umg.pdf", dpi=600)
# plt.show()
plt.clf()
plt.close()

# ***********************************
# ***********************************
# ***********************************

# plt.figure(figsize=(8, 8))
# ax = plt.gca()

# bins = 125

# plt.hist(u, bins=bins, density=True, histtype='step', color='b', label='u')
# plt.hist(g, bins=bins, density=True, histtype='step', color='g', label='g')
# plt.hist(r, bins=bins, density=True, histtype='step', color='r', label='r')
# plt.hist(i, bins=bins, density=True, histtype='step', color='gray', label='i')
# plt.hist(z, bins=bins, density=True, histtype='step', color='k', label='z')

# ax.set_xlabel("$m$")
# ax.set_ylabel("Normalized Count")
# ax.set_xlim([15, 27.5])
# ax.set_ylim([0, 0.8])

# ax.legend(loc='best', frameon=False)

# ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

# # plt.tight_layout()

# plt.show()
# plt.clf()
# plt.close()

# ***********************************
# ***********************************
# ***********************************

# dM_index = np.where((maintype == "M") & (M_G >= 5.0))[0]

# r = prop[dM_index]['rmag_SDSSDR12']
# percent_above = (prop[dM_index]['ZTF_g_nabove'] / prop[dM_index]['ZTF_g_ngood']) * 100
# Halpha_EQW = prop[dM_index]['Halpha_EqW']
# chisq = np.log10(prop[dM_index]['ZTF_g_Chi2'])

# Halpha_EQW_index  = Halpha_EQW < 10.0

# plt.rc('font', size=15)          # controls default text sizes
# plt.rc('axes', titlesize=15)     # fontsize of the axes title
# plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
# plt.rc('legend', fontsize=12)    # legend fontsize
# plt.rc('figure', titlesize=12)  # fontsize of the figure title

# cmap = plt.get_cmap('seismic')

# plt.figure(figsize=(9, 8))
# ax = plt.gca()
# sc = plt.scatter(chisq[Halpha_EQW_index], percent_above[Halpha_EQW_index], c=Halpha_EQW[Halpha_EQW_index], s=5.0, cmap=cmap, vmin=-20, vmax=10)

# divider1 = make_axes_locatable(ax)
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = plt.colorbar(sc, cax=cax1)  # , ticks=np.arange(letterSpt.size))
# cbar1.ax.get_yaxis().labelpad = 10
# cbar1.ax.set_ylabel(r'H$\alpha$ EQW', rotation=270)
# # cbar1.ax.set_yticklabels(np.flip(letterSpt_forlabel))

# ax.set_xlabel("log$\chi^2$")
# # ax.set_xlabel("$r [mag]$")
# ax.set_ylabel("Percent Above [\%]")
# # ax.set_xlim([-1.0, 5.0])
# # ax.set_ylim([-0.5, 2.0])

# # ax.invert_yaxis()

# # ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
# # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
# # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# plt.tight_layout()
# # plt.savefig("gmr_umg.pdf", dpi=600)
# plt.show()
# plt.clf()
# plt.close()

# ***********************************
# ***********************************
# ***********************************

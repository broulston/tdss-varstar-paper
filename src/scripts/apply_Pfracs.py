import numpy as np
from astropy.table import Table, hstack, vstack
import Tools
from tqdm import trange

import paths

###############################################################################
###############################################################################
###############################################################################
# Read in ZTF LC data

ZTF_LC_dir = paths.data

ZTF_filters = ["g", "r"]
ZTF_LC_file_names = [
    f"TDSS_VarStar_ZTFDR6_{ZTF_filter}_GroupID.fits" for ZTF_filter in ZTF_filters
]
ZTF_g_LCs = Table.read(ZTF_LC_dir / ZTF_LC_file_names[0])
ZTF_r_LCs = Table.read(ZTF_LC_dir / ZTF_LC_file_names[1])

###############################################################################
###############################################################################
###############################################################################
#  Read in TDSS Prob that was decided as VAR, len=11,654
TDSS_prop_isVar = Table.read(
    paths.data / "Var_notVar_data/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS.fits"
)

#  Read in TDSS Prob that was decided as NOT VAR, len=13,467
TDSS_prop_isNOTVar = Table.read(
    paths.data / "Var_notVar_data/TDSS_VarStar_FINAL_NonVar_ALL_PROP_STATS.fits"
)

###############################################################################
###############################################################################
###############################################################################
# Read in the old (P > 0.1d) Pfrac table
Pfrac_table_Pgt01d = Table.read(paths.static / "TDSS_Var_Pfrac_Table.fits")

# Add extra columns that we added for the P < 0.1d search
ExtraHighFreqPower = np.array([False] * len(Pfrac_table_Pgt01d))
t0_index = Pfrac_table_Pgt01d.index_column("ZTF_g_Comments") + 1
Pfrac_table_Pgt01d.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_g_ExtraHighFrequencyPower"
)
t0_index = Pfrac_table_Pgt01d.index_column("ZTF_r_Comments") + 1
Pfrac_table_Pgt01d.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_r_ExtraHighFrequencyPower"
)

is_periodic_Pgt01d = (TDSS_prop_isVar["ZTF_g_logProb"] <= -5).data | (
    TDSS_prop_isVar["ZTF_r_logProb"] <= -5
).data
is_periodic_index_Pgt01d = np.where(is_periodic_Pgt01d)[0]

# Read in the new (P < 0.1d) Pfrac table
Pfrac_table_Plt01d = Table.read(paths.static / "TDSS_Var_Pfrac_Table_Plt01d.fits")


###############################################################################
###############################################################################
###############################################################################
# Read in the index table to go from Full (25121) to the Var selected (11654)
full_index_to_isVar = Table.read(paths.data / "Var_notVar_data/full_index_table.fits")
# Make connecting array.
isVar_index_to_full_index = np.where(full_index_to_isVar["full_index"])[
    0
]  # len = 11654, max() --> 25120

isNOTVAR_index_to_full_index = np.where(~full_index_to_isVar["full_index"].data)[0]

# Connect previous P > 0.1d index to Full index
isVar_index_to_full_Pgt01d = isVar_index_to_full_index[is_periodic_index_Pgt01d]

# Read in the LCprops files for the P < 0.1d sample
TDSS_P_prop = Table.read(paths.static / "TDSS_VarStar_LCprops_02-10-2023_Plt01d.fits")
###############################################################################
###############################################################################
###############################################################################
# This was the FAP limit used throughout
log10FAP = -5

# Find where the P < 0.1d sample is periodic, should be len=1074, with index into Full table
is_one_periodic = (
    ((TDSS_P_prop["ZTF_g_logProb"] <= log10FAP) * (TDSS_P_prop["ZTF_g_P"] <= 0.1))
    | ((TDSS_P_prop["ZTF_r_logProb"] <= log10FAP) * (TDSS_P_prop["ZTF_r_P"] <= 0.1))
).data  # & TDSS_isVar_selection["full_index"]
is_one_periodic_index = np.where(is_one_periodic == True)[0]

Pfrac_table_Pgt01d.add_column(isVar_index_to_full_Pgt01d, index=0, name="full_index")
Pfrac_table_Plt01d.add_column(is_one_periodic_index, index=0, name="full_index")

Pfrac_table_Pgt01d.add_column(
    np.zeros(len(Pfrac_table_Pgt01d)).astype(bool), index=0, name="fromNew"
)
Pfrac_table_Plt01d.add_column(
    np.ones(len(Pfrac_table_Plt01d)).astype(bool), index=0, name="fromNew"
)

Pfrac_table_Pgt01d["ZTF_g_Comments"].fill_value = ""
Pfrac_table_Pgt01d["ZTF_r_Comments"].fill_value = ""

Pfrac_table_Plt01d["ZTF_g_Comments"].fill_value = ""
Pfrac_table_Plt01d["ZTF_r_Comments"].fill_value = ""

combined_Pfrac_table = Pfrac_table_Pgt01d.copy()

new_to_add_full_index = []
for ii in range(len(Pfrac_table_Plt01d)):
    this_full_index_Plt01d = is_one_periodic_index[ii]
    # P < 0.1d object is already in the isVar sample! Need to check if it already as Vi'd
    if this_full_index_Plt01d in isVar_index_to_full_index:
        # object is already in the previous P sample and has been Vi'd. Will need to check which one to keep.....
        if this_full_index_Plt01d in isVar_index_to_full_Pgt01d:
            where_to_insert_PFrac = np.where(
                isVar_index_to_full_Pgt01d == this_full_index_Plt01d
            )[0][0]

            oldVi_g_isP, oldVi_r_isP = Pfrac_table_Pgt01d[where_to_insert_PFrac][
                ["ZTF_g_isP", "ZTF_r_isP"]
            ]
            oldVi_g_P, oldVi_r_P = TDSS_prop_isVar[
                is_periodic_index_Pgt01d[where_to_insert_PFrac]
            ][["ZTF_g_P", "ZTF_r_P"]]
            oldVi_g_logProb, oldVi_r_logProb = TDSS_prop_isVar[
                is_periodic_index_Pgt01d[where_to_insert_PFrac]
            ][["ZTF_g_logProb", "ZTF_r_logProb"]]

            newVi_g_isP, newVi_r_isP = Pfrac_table_Plt01d[ii][
                ["ZTF_g_isP", "ZTF_r_isP"]
            ]
            newVi_g_P, newVi_r_P = TDSS_P_prop[this_full_index_Plt01d][
                ["ZTF_g_P", "ZTF_r_P"]
            ]
            newVi_g_logProb, newVi_r_logProb = TDSS_P_prop[this_full_index_Plt01d][
                ["ZTF_g_logProb", "ZTF_r_logProb"]
            ]

            old_is_any_p = np.any([oldVi_g_isP, oldVi_r_isP])
            new_is_any_p = np.any([newVi_g_isP, newVi_r_isP])
            # np.ma.is_masked(g_logProb)

            # always want to combine trends, and comments
            combined_Pfrac_table[where_to_insert_PFrac][
                "ZTF_g_LongTermTrends"
            ] = np.any(
                [
                    Pfrac_table_Pgt01d[where_to_insert_PFrac]["ZTF_g_LongTermTrends"],
                    Pfrac_table_Plt01d[ii]["ZTF_g_LongTermTrends"],
                ]
            )
            combined_Pfrac_table[where_to_insert_PFrac][
                "ZTF_r_LongTermTrends"
            ] = np.any(
                [
                    Pfrac_table_Pgt01d[where_to_insert_PFrac]["ZTF_r_LongTermTrends"],
                    Pfrac_table_Plt01d[ii]["ZTF_r_LongTermTrends"],
                ]
            )

            combined_Pfrac_table[where_to_insert_PFrac]["ZTF_g_Comments"] = (
                Pfrac_table_Pgt01d["ZTF_g_Comments"].filled()[where_to_insert_PFrac]
                + " "
                + Pfrac_table_Plt01d["ZTF_g_Comments"].filled()[ii]
            )
            combined_Pfrac_table[where_to_insert_PFrac]["ZTF_r_Comments"] = (
                Pfrac_table_Pgt01d["ZTF_r_Comments"].filled()[where_to_insert_PFrac]
                + " "
                + Pfrac_table_Plt01d["ZTF_r_Comments"].filled()[ii]
            )

            combined_Pfrac_table[where_to_insert_PFrac][
                "ZTF_g_ExtraHighFrequencyPower"
            ] = np.any(
                [
                    Pfrac_table_Pgt01d[where_to_insert_PFrac][
                        "ZTF_g_ExtraHighFrequencyPower"
                    ],
                    Pfrac_table_Plt01d[ii]["ZTF_g_ExtraHighFrequencyPower"],
                ]
            )
            combined_Pfrac_table[where_to_insert_PFrac][
                "ZTF_r_ExtraHighFrequencyPower"
            ] = np.any(
                [
                    Pfrac_table_Pgt01d[where_to_insert_PFrac][
                        "ZTF_r_ExtraHighFrequencyPower"
                    ],
                    Pfrac_table_Plt01d[ii]["ZTF_r_ExtraHighFrequencyPower"],
                ]
            )
            #  old is periodic but not new, so use old and combine comments etc
            if old_is_any_p and not new_is_any_p:
                pass
            #  new is periodic but not old, so use new and combine comments etc
            elif not old_is_any_p and new_is_any_p:
                combined_Pfrac_table[where_to_insert_PFrac] = Pfrac_table_Plt01d[ii]
            # neither old or new are periodic, so use old and combine comments etc
            elif not old_is_any_p and not new_is_any_p:
                pass
            #  both old and new arre periodic need to determine which to keep
            else:
                # new has 1 P sig <-5 and P < 0.1 so use that
                if np.any(
                    (np.array([newVi_g_P, newVi_r_P]) < 0.1)
                    & (np.array([newVi_g_logProb, newVi_r_logProb]) < -5)
                ):
                    combined_Pfrac_table[where_to_insert_PFrac] = Pfrac_table_Plt01d[ii]
                else:  # other wise if no sig P < 0.1d use old!
                    pass
        else:  # object was not previois P, so is new P! So just use the new Vi to adjust the P features as before
            combined_Pfrac_table.add_row(Pfrac_table_Plt01d[ii])
            # new_to_add_full_index.append(this_full_index_Plt01d)
    else:  # this object is new! so can just take the raw information that is in the new P < 0.1d Vi
        combined_Pfrac_table.add_row(Pfrac_table_Plt01d[ii])
        new_to_add_full_index.append(this_full_index_Plt01d)

new_to_add_full_index = np.array(new_to_add_full_index)

new_TDSS_prop = TDSS_prop_isVar.copy()
len_new_table = len(TDSS_prop_isVar)  # + len(new_to_add_full_index)

is_P = np.array([False] * len_new_table)
LongTermTrends = np.array([False] * len_new_table)
VarType = np.array([" " * 100] * len_new_table)
Comments = np.array([" " * 100] * len_new_table)
ExtraHighFreqPower = np.array([False] * len_new_table)

t0_index = new_TDSS_prop.index_column("ZTF_g_t0") + 1
new_TDSS_prop.add_column(is_P, index=t0_index, name="ZTF_g_is_P")
t0_index = new_TDSS_prop.index_column("ZTF_g_t0") + 1
new_TDSS_prop.add_column(VarType, index=t0_index, name="ZTF_g_ViVarType")
t0_index = new_TDSS_prop.index_column("ZTF_g_t0") + 1
new_TDSS_prop.add_column(LongTermTrends, index=t0_index, name="ZTF_g_LongTermTrends")
t0_index = new_TDSS_prop.index_column("ZTF_g_t0") + 1
new_TDSS_prop.add_column(Comments, index=t0_index, name="ZTF_g_ViComments")
t0_index = new_TDSS_prop.index_column("ZTF_g_t0") + 1
new_TDSS_prop.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_g_ExtraHighFrequencyPower"
)

t0_index = new_TDSS_prop.index_column("ZTF_r_t0") + 1
new_TDSS_prop.add_column(is_P, index=t0_index, name="ZTF_r_is_P")
t0_index = new_TDSS_prop.index_column("ZTF_r_t0") + 1
new_TDSS_prop.add_column(VarType, index=t0_index, name="ZTF_r_ViVarType")
t0_index = new_TDSS_prop.index_column("ZTF_r_t0") + 1
new_TDSS_prop.add_column(LongTermTrends, index=t0_index, name="ZTF_r_LongTermTrends")
t0_index = new_TDSS_prop.index_column("ZTF_r_t0") + 1
new_TDSS_prop.add_column(Comments, index=t0_index, name="ZTF_r_ViComments")
t0_index = new_TDSS_prop.index_column("ZTF_r_t0") + 1
new_TDSS_prop.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_r_ExtraHighFrequencyPower"
)

is_P = np.array([False] * len(TDSS_prop_isNOTVar))
LongTermTrends = np.array([False] * len(TDSS_prop_isNOTVar))
VarType = np.array([" " * 100] * len(TDSS_prop_isNOTVar))
Comments = np.array([" " * 100] * len(TDSS_prop_isNOTVar))
ExtraHighFreqPower = np.array([False] * len(TDSS_prop_isNOTVar))

t0_index = TDSS_prop_isNOTVar.index_column("ZTF_g_t0") + 1
TDSS_prop_isNOTVar.add_column(is_P, index=t0_index, name="ZTF_g_is_P")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_g_t0") + 1
TDSS_prop_isNOTVar.add_column(VarType, index=t0_index, name="ZTF_g_ViVarType")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_g_t0") + 1
TDSS_prop_isNOTVar.add_column(
    LongTermTrends, index=t0_index, name="ZTF_g_LongTermTrends"
)
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_g_t0") + 1
TDSS_prop_isNOTVar.add_column(Comments, index=t0_index, name="ZTF_g_ViComments")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_g_t0") + 1
TDSS_prop_isNOTVar.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_g_ExtraHighFrequencyPower"
)

t0_index = TDSS_prop_isNOTVar.index_column("ZTF_r_t0") + 1
TDSS_prop_isNOTVar.add_column(is_P, index=t0_index, name="ZTF_r_is_P")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_r_t0") + 1
TDSS_prop_isNOTVar.add_column(VarType, index=t0_index, name="ZTF_r_ViVarType")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_r_t0") + 1
TDSS_prop_isNOTVar.add_column(
    LongTermTrends, index=t0_index, name="ZTF_r_LongTermTrends"
)
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_r_t0") + 1
TDSS_prop_isNOTVar.add_column(Comments, index=t0_index, name="ZTF_r_ViComments")
t0_index = TDSS_prop_isNOTVar.index_column("ZTF_r_t0") + 1
TDSS_prop_isNOTVar.add_column(
    ExtraHighFreqPower, index=t0_index, name="ZTF_r_ExtraHighFrequencyPower"
)

new_Plt01d_colnames_to_replace = TDSS_P_prop.colnames[3:]


def apply_Pfrac(new_TDSS_prop, where_proptable, ROW_pfrac, TDSS_P_prop, fromNew):
    if fromNew:
        for this_col in new_Plt01d_colnames_to_replace:
            new_TDSS_prop[where_proptable][this_col] = TDSS_P_prop[
                ROW_pfrac["full_index"]
            ][this_col]

    ROW_P = new_TDSS_prop[where_proptable]

    new_TDSS_prop[where_proptable]["ZTF_g_P"] = (
        ROW_P["ZTF_g_P"] * ROW_pfrac["ZTF_g_Pfrac"]
    )
    new_TDSS_prop[where_proptable]["ZTF_g_is_P"] = ROW_pfrac["ZTF_g_isP"]
    new_TDSS_prop[where_proptable]["ZTF_r_P"] = (
        ROW_P["ZTF_r_P"] * ROW_pfrac["ZTF_r_Pfrac"]
    )
    new_TDSS_prop[where_proptable]["ZTF_r_is_P"] = ROW_pfrac["ZTF_r_isP"]

    new_TDSS_prop[where_proptable]["ZTF_g_ViVarType"] = ROW_pfrac["ZTF_g_VarType"]
    new_TDSS_prop[where_proptable]["ZTF_r_ViVarType"] = ROW_pfrac["ZTF_r_VarType"]

    new_TDSS_prop[where_proptable]["ZTF_g_ViComments"] = ROW_pfrac["ZTF_g_Comments"]
    new_TDSS_prop[where_proptable]["ZTF_r_ViComments"] = ROW_pfrac["ZTF_r_Comments"]

    new_TDSS_prop[where_proptable]["ZTF_g_LongTermTrends"] = ROW_pfrac[
        "ZTF_g_LongTermTrends"
    ]
    new_TDSS_prop[where_proptable]["ZTF_r_LongTermTrends"] = ROW_pfrac[
        "ZTF_r_LongTermTrends"
    ]

    new_TDSS_prop[where_proptable]["ZTF_g_ExtraHighFrequencyPower"] = ROW_pfrac[
        "ZTF_g_ExtraHighFrequencyPower"
    ]
    new_TDSS_prop[where_proptable]["ZTF_r_ExtraHighFrequencyPower"] = ROW_pfrac[
        "ZTF_r_ExtraHighFrequencyPower"
    ]

    isZTFg = (
        False
        if np.ma.is_masked(ROW_P["ZTF_g_Nepochs"])
        else (ROW_P["ZTF_g_Nepochs"] > 10)
    )
    isZTFr = (
        False
        if np.ma.is_masked(ROW_P["ZTF_r_Nepochs"])
        else (ROW_P["ZTF_r_Nepochs"] > 10)
    )

    if isZTFg:
        ZTF_g_lc_data = ZTF_g_LCs[(ZTF_g_LCs["ZTF_GroupID"] == ROW_P["ZTF_GroupID"])][
            "mjd", "mag", "magerr"
        ]
        lc_data, LC_stat_properties = Tools.process_LC(
            ZTF_g_lc_data.copy(), fltRange=5.0
        )

        goodQualIndex = np.where(lc_data["QualFlag"] == True)[0]
        mjd = lc_data["mjd"][goodQualIndex].data
        mag = lc_data["mag"][goodQualIndex].data
        err = np.abs(lc_data["magerr"][goodQualIndex].data)

        period = new_TDSS_prop[where_proptable]["ZTF_g_P"]
        AFD_data = Tools.AFD([mjd, mag, err], period, alpha=0.99, Nmax=6)
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mfit = AFD_data
        Amp = y_fit.max() - y_fit.min()
        t0 = (mjd - (phased_t * period)).max()

        new_TDSS_prop[where_proptable]["ZTF_g_Amp"] = Amp
        new_TDSS_prop[where_proptable]["ZTF_g_t0"] = t0

    if isZTFr:
        ZTF_r_lc_data = ZTF_r_LCs[(ZTF_r_LCs["ZTF_GroupID"] == ROW_P["ZTF_GroupID"])][
            "mjd", "mag", "magerr"
        ]
        lc_data, LC_stat_properties = Tools.process_LC(
            ZTF_r_lc_data.copy(), fltRange=5.0
        )

        goodQualIndex = np.where(lc_data["QualFlag"] == True)[0]
        mjd = lc_data["mjd"][goodQualIndex].data
        mag = lc_data["mag"][goodQualIndex].data
        err = np.abs(lc_data["magerr"][goodQualIndex].data)

        period = new_TDSS_prop[where_proptable]["ZTF_r_P"]
        AFD_data = Tools.AFD([mjd, mag, err], period, alpha=0.99, Nmax=6)
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mfit = AFD_data
        Amp = y_fit.max() - y_fit.min()
        t0 = (mjd - (phased_t * period)).max()

        new_TDSS_prop[where_proptable]["ZTF_r_Amp"] = Amp
        new_TDSS_prop[where_proptable]["ZTF_r_t0"] = t0


for ii in trange(len(combined_Pfrac_table)):
    ROW_pfrac = combined_Pfrac_table[ii]
    this_full_index = ROW_pfrac["full_index"]

    fromNew = ROW_pfrac["fromNew"]

    # this is new so need to add row at end and then do P update
    if this_full_index in new_to_add_full_index:
        NOTVAR_prop_ROW_index = np.where(
            isNOTVAR_index_to_full_index == this_full_index
        )[0][0]
        new_TDSS_prop.add_row(TDSS_prop_isNOTVar[NOTVAR_prop_ROW_index])

        where_proptable = -1
        apply_Pfrac(new_TDSS_prop, where_proptable, ROW_pfrac, TDSS_P_prop, fromNew)
    else:  # already in the table, so find where and do the P frac update!!
        where_proptable = np.where(isVar_index_to_full_index == this_full_index)[0][0]
        # new_TDSS_prop[where_proptable]
        apply_Pfrac(new_TDSS_prop, where_proptable, ROW_pfrac, TDSS_P_prop, fromNew)

new_TDSS_prop.sort("ra_GaiaEDR3")
new_TDSS_prop.write(
    paths.data / "TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits",
    format="fits",
    overwrite=True,
)

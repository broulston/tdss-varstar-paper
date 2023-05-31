rule all:
    input:
        "src/data/Var_notVar_data",
        "src/data/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits"

rule makeVarnotVarData:
    input: 
        "src/static/TDSS_SES+PREV_DR16DR12griLT20_GaiaEDR3_Drake2014PerVar_CSSID_ZTFIDs_LCpointer_PyHammer_EqW_LCProps.fits",
        "src/static/VarStar_LCSTATS2.fits",
        "src/static/VarStar_rawSigmaAboveBelow.fits",
        "src/static/VarStar_Controljittered_LCSTATS_withP2.fits",
        "src/static/VarStar_rawSigmaAboveBelow.fits",
        "src/scripts/make_Var_notVar_data.ipynb"
    output:
        directory("src/data/Var_notVar_data")
    conda:
        "environment.yml"
    cache:
        True
    shell:
        "jupyter-execute src/scripts/make_Var_notVar_data.ipynb"

rule apply_Pfracs:
    input:
        "src/data/Var_notVar_data",
        "src/static/TDSS_Var_Pfrac_Table.fits",
        "src/static/TDSS_Var_Pfrac_Table_Plt01d.fits",
        "src/static/TDSS_VarStar_LCprops_02-10-2023_Plt01d.fits",
        "src/data/TDSS_VarStar_ZTFDR6_g_GroupID.fits",
        "src/data/TDSS_VarStar_ZTFDR6_r_GroupID.fits"
    output:
        "src/data/TDSS_VarStar_FINAL_Var_ALL_PROP_STATS_dealiased.fits"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/apply_Pfracs.py"

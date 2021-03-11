#!/bin/bash
# A Bash script to run calc_hdx.py

python $HDXER_PATH/HDXer/calc_hdx.py -t $BPTI_PATH/BPTI_simulations/Run_1/bpti_5pti_reimg_protonly.xtc $BPTI_PATH/BPTI_simulations/Run_2/bpti_5pti_reimg_protonly.xtc $BPTI_PATH/BPTI_simulations/Run_3/bpti_5pti_reimg_protonly.xtc $BPTI_PATH/BPTI_simulations/Run_4/bpti_5pti_reimg_protonly.xtc $BPTI_PATH/BPTI_simulations/Run_5/bpti_5pti_reimg_protonly.xtc -p $BPTI_PATH/BPTI_simulations/Run_1/bpti_5pti_eq6_protonly.gro -m BestVendruscolo -log BPTI_calc_hdx.log -out BPTI_ -exp $BPTI_PATH/RTB_pD_7.4/expt_data/BPTI_expt_dfracs.dat -seg $BPTI_PATH/RTB_pD_7.4/expt_data/BPTI_residue_segs.txt -mopt "{ 'save_detailed' : True }"
 

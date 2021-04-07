#!/bin/bash
# A Bash script to run calc_hdx.py

python $HDXER_PATH/HDXer/calc_hdx.py -t $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_1/bpti_5pti_reimg_protonly.xtc $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_2/bpti_5pti_reimg_protonly.xtc $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_3/bpti_5pti_reimg_protonly.xtc $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_4/bpti_5pti_reimg_protonly.xtc $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_5/bpti_5pti_reimg_protonly.xtc -p $HDXER_PATH/tutorials/BPTI/BPTI_simulations/Run_1/bpti_5pti_eq6_protonly.gro -m BestVendruscolo -log BPTI_calc_hdx.log -out BPTI_ -exp $HDXER_PATH/tutorials/BPTI/BPTI_expt_data/BPTI_expt_dfracs.dat -seg $HDXER_PATH/tutorials/BPTI/BPTI_expt_data/BPTI_residue_segs.txt -mopt "{ 'save_detailed' : True }"
 

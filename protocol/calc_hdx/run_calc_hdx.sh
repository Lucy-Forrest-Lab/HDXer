#!/bin/bash
# A Bash script to run calc_hdx.py

python $HDXER_PATH/HDXer/calc_hdx.py -t $BPTI_PATH/structural_ensembles/BPTI_full.dcd -p $BPTI_PATH/structural_ensembles/BPTI_full.pdb -m Radou -log BPTI.log -out BPTI_ -exp $BPTI_PATH/expt_data/BPTI_expt_dfracs.dat -seg $BPTI_PATH/expt_data/BPTI_residue_segs.txt -mopt "{ 'save_detailed' : True }"

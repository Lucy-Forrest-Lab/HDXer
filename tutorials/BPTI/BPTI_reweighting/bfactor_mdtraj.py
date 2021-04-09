#!/usr/bin/env python

# Script to insert atom-based b-factors into PDB with MDtraj
# (or equally to write out atom/residue b-factors to file)

import numpy as np
import mdtraj as md

# Load in PDB and desired residue segments
inpdb = md.load("5pti.pdb")
df_segs_diffs = np.loadtxt("Reweighted-Predicted_diffs.dat", dtype=[ ('segs', np.int32, (1,)),\
                           ('fracs', np.float64, (4,))])

#Atom based bfacs = list of lists
bfacs = []
for i in range(inpdb.n_atoms):
    bfacs.append([])
for atm, dfracs in enumerate(bfacs):
    for line, seg in enumerate(df_segs_diffs['segs'], start=0):
        if all((inpdb.topology.atom(atm).residue.resSeq == seg)):
            dfracs.append(df_segs_diffs['fracs'][line, 3]) # 120.0 min
        else:
            continue

# convert to numpy, averaging all the lists
# This will throw up a warning for nans
bfac_array = np.asarray([ np.mean(np.asarray(x)) for x in bfacs ])
bfac_array = np.nan_to_num(bfac_array)

np.savetxt("Reweighted-Predicted_byatom_1min.txt", bfac_array)
inpdb.save_pdb("Reweighted-Predicted_byatom_1min.pdb", bfactors=bfac_array)

#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import pickle






# Define variables from args

folderlist = args.folders
expt_path = args.expfile
update_lambdas = args.do_reweight
optimize_params = args.do_params
mc_opt = args.do_mc
mc_sample_params = args.do_mcsmpl
kintfile = args.ratefile
times = args.times
prefix = args.outprefix
Bc = args.bc
Bh = args.bh
Bcrange = args.bcrange
Bhrange = args.bhrange
chisquareref=args.refvar
nequil=args.equilsteps
gamma = args.gamma
tol = args.tolerance
count = args.maxiters
paramcount = args.parammaxiters
ratef = args.ratef
rfact = args.scaleff
paramsfact = args.paramf
randomweights = args.randomweights
T = args.temp
kT=0.008314598*T
rstfile = args.restart

if rstfile is not None:
    restart = True
else:
    restart = False
#    overwrite = False

if mc_sample_params:
    optimize_params = False
    mc_opt = False

# Sorting key for contacts/Hbonds files
def strip_filename(fn, extrastr=""):
    """Sorting key that will strip the integer residue number
       from a Contacts/Hbonds filename. Expects filenames of 
       the sort 'Contacts_123.tmp' - splits on _ and . 

       If filenames are not quite of this format, optionally
       the 'extrastr' argument can be used to add an additional
       string to the filename, e.g. 'Contacts_chain_0_res_123.tmp'"""
    try:
        _ = fn.split("Contacts_"+extrastr)[1]
        return int(_.split(".")[0])
    except:
        pass
    try:
        _ = fn.split("Hbonds_"+extrastr)[1]
        return int(_.split(".")[0])
    except:
        raise NameError("File found without correct 'Contacts_' or 'Hbonds_' format filename: %s" % fn)

# Read in single column datafiles        
def files_to_array(fnames):
    """Read in data fom list of files with np.loadtxt. 
       Returns array of shape (n_files, n_data_per_file)"""
    l = [ np.loadtxt(f) for f in fnames ]
    try:
        return np.stack(l, axis=0)
    except ValueError:
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")

# Read in initial contacts & H-bonds.
# Store as 2D-np arrays of shape (n_residues, n_frames)

# Treat separate chains as separate frames. Can be upweighted/downweighted individually
# (Only applicable if we add extra lines here to read in the extra Contacts/Hbonds files for each chain)
def read_contacts_hbonds(folderlist):
    """Read in contact & hbond files from defined folders & return as numpy arrays,
       along with the resids they correspond to.
       
       Usage: read_contacts_hbonds(folderlist)

       Returns: contacts[n_residues, n_frames], hbonds[n_residues, n_frames], sorted_resids"""
 
    contactfiles, hbondfiles = [],[]
    for folder in folderlist:
        contactfiles.append(sorted(glob(os.path.join(folder, "Contacts_*.tmp")), key=lambda x: strip_filename(x)))
        hbondfiles.append(sorted(glob(os.path.join(folder, "Hbonds_*.tmp")), key=lambda x: strip_filename(x)))
    
    resids = []
    # This is a list comprehension with the try/except for the extra strings
    for curr in contactfiles:        
        _ = []
        for f in curr:
            try:
                _.append( strip_filename(f) ) 
            except NameError:
                _.append( strip_filename(f) ) # E.g. for 2 chain system
        resids.append(_)
    
    sorted_resids = deepcopy(resids)
    sorted_resids.sort(key=lambda _: len(_))
    filters = list(map(lambda _: np.in1d(_, sorted_resids[0]), resids)) # Get indices to filter by shortest
    new_resids = []
    for r, f in list(zip(resids, filters)):
        new_resids.append(np.array(r)[f])
    new_resids = np.stack(new_resids)
    if not np.diff(new_resids, axis=0).sum(): # If sum of differences between filtered resids == 0
        pass
    else:
        raise ValueError("Error in filtering trajectories to common residues - do residue IDs match up in your intrinsic rate files?")
    
    _contacts = list(map(lambda x, y: x[y], [ files_to_array(curr_cfiles) for curr_cfiles in contactfiles ], filters))
    _hbonds = list(map(lambda x, y: x[y], [ files_to_array(curr_hfiles) for curr_hfiles in hbondfiles ], filters))
    
    
    contacts = np.concatenate(_contacts, axis=1)
    print("Contacts read")
    hbonds = np.concatenate(_hbonds, axis=1)
    print("Hbonds read")
    assert (contacts.shape == hbonds.shape)
    return contacts, hbonds, sorted_resids



# Read intrinsic rates, multiply by times
def read_kints_segments(kintfile, expt_path, n_res, times, sorted_resids):
    """Read in intrinsic rates, segments, and expt deuterated fractions.
       All will be reshaped into 3D arrays of [n_segments, n_residues, n_times]

       Requires number of residues, times, and a list of residue IDs 
       for which contacts/H-bonds have been read  in

       Usage: read_kints_segments(kintfile, expt_path, n_res, times, sorted_resids)

       Returns: kint, expt_dfrac, segfilters (all of shape [n_segments, n_residues, n_times])"""

    kint = np.loadtxt(kintfile, usecols=(1,)) # We only need one file here and it'll be filtered based on its residue IDs
    kintresid = np.loadtxt(kintfile, usecols=(0,))
    kintfilter = np.in1d(kintresid, sorted_resids[0])
    kint = kint[kintfilter]
    final_resid = kintresid[kintfilter]
    kint = np.repeat(kint[:, np.newaxis], len(times), axis=1)*times # Make sure len(times) is no. of expt times
    # Read deuterated fractions, shape will be (n_residues, n_times)
    exp_dfrac = np.loadtxt(expt_path, usecols=tuple(range(2,2+len(times))))
    segments = np.loadtxt(expt_path, usecols=(0,1), dtype=np.int32)
    
    # convert expt to (segments, residues, times)
    exp_dfrac = exp_dfrac[:,np.newaxis,:].repeat(n_res, axis=1)
    # convert kint to (segments, residues, times)
    kint = kint[np.newaxis,:,:].repeat(len(segments), axis=0)
    
    # Make a set of filters that defines the residues in each segment & timepoint
    segfilters=[]
    for seg in segments:
        seg_resids = range(seg[0], seg[1]+1)
        segfilters.append(np.in1d(final_resid, seg_resids[1:])) # Filter but skipping first residue in segment
    segfilters = np.array(segfilters)
    segfilters = np.repeat(segfilters[:, :, np.newaxis], len(times), axis=2) # Repeat to shape (n_segments, n_residues, n_times)
    
    assert all((segfilters.shape == exp_dfrac.shape, \
               segfilters.shape == kint.shape, \
               n_res == len(final_resid))) # Check we've at least read in the right number!
    
    print("Segments and experimental dfracs read")
    return kint, exp_dfrac, segfilters


# Setup functions
def setup_no_restart():
    """Setup initial variables when a restart file is NOT read in"""
    global contacts, hbonds, kint, exp_dfrac, segfilters, lambdas, nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, currcount
    contacts, hbonds, sorted_resids = read_contacts_hbonds(folderlist)
    nresidues=len(hbonds)
    kint, exp_dfrac, segfilters = read_kints_segments(kintfile, expt_path, nresidues, times, sorted_resids) 
    # Starting lambda values
    lambdas=np.zeros(nresidues)
    # if optionally average over parameters, define _lambdanewaverh and _lambdanewaverc

#    if mc_sample_params:
    global _lambdanew, chisquareav, _lambdanewaverh, _lambdanewaverc, _lambdanewh, _lambdanewc, lambdasc, lambdash
    _lambdanewaverh=np.zeros(nresidues)
    _lambdanewaverc=np.zeros(nresidues)
    _lambdanew=np.zeros(nresidues) 
    _lambdanewh=np.zeros(nresidues)
    _lambdanewc=np.zeros(nresidues)
    lambdasc=np.zeros(nresidues)
    lambdash=np.zeros(nresidues)

    chisquareav=0

    # Write initial parameter values
    nframes = contacts.shape[1]
    with open("%sinitial_params.dat" % prefix, 'w') as f:
        f.write("Temp, kT, convergence tolerance, BetaC, BetaH, gamma, update rate (step size) factor, nframes\n")
        f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n" % (T, kT, tol, Bc, Bh, gamma, ratef, nframes))

    # Random initial weights perturb the ensemble - can we recreate the original trajectory with equally weighted frames?
    # Choose to either generate new random weights or set random seed for reproducibility (e.g. set to 1234)
    #np.random.seed(1234)
    if randomweights:
        np.random.seed(1234)
        iniweights=np.random.rand(nframes)
        np.savetxt("ini_weights_uniform_randseed1234.dat", iniweights)
        np.random.seed()
    else:
        iniweights=np.ones(nframes)

    with open("%swork.dat" % prefix, 'w') as f:
        f.write("# gamma, chisquare, work(kJ/mol)\n")
    with open("%sper_iteration_output.dat" % prefix, 'w') as f:
        f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc \n")
    with open("%sper_restart_output.dat" % prefix, 'w') as f:
        f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc, work \n")

    # Set some constants & initial values for the iteration loop
    num = -kint*segfilters
    exp_df_segf = exp_dfrac*segfilters
    normseg = np.sum(segfilters)
    numseg = segfilters.shape[0]
    lambdamod = 0.0
    deltalambdamod = 0.0
    currcount = 0


def setup_restart(rstfile):
    """Setup initial variables when a restart file IS read in"""
    # List of pickle variables:
    global contacts, hbonds, kint, exp_dfrac, segfilters, lambdas, \
           lambdasc, lambdash, _lambdanewaverh, _lambdanewaverc, \
           _lambdanew, _lambdanewh, _lambdanewc, chisquareav, \
           nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, \
           Bc, Bh, gamma, ratef, avesigmalnpi, currcount
    contacts, hbonds, kint, exp_dfrac, segfilters, lambdas,lambdasc ,lambdash , _lambdanewaverh, _lambdanewaverc, \
    _lambdanew, _lambdanewh, _lambdanewc, chisquareav, \
    nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, \
    Bc, Bh, gamma, ratef, avesigmalnpi, currcount = pickle.load(open(rstfile, 'rb'))
    # Append to existing files
    with open("%sinitial_params.dat" % prefix, 'a') as f:
        f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
        f.write("Temp, kT, convergence tolerance, BetaC, BetaH, gamma, update rate (step size) factor, nframes\n")
        f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n" % (T, kT, tol, Bc, Bh, gamma, ratef, nframes))
    with open("%swork.dat" % prefix, 'a') as f:
        f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
        f.write("# gamma, chisquare, work(kJ/mol)\n")
    with open("%sper_iteration_output.dat" % prefix, 'a') as f:
        f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
        f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc \n")
    with open("%sper_restart_output.dat" % prefix, 'a') as f:
        f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
        f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc, work \n")
    print("Restart file %s read" % rstfile)

# Do setup:
if restart:
    setup_restart(rstfile)
else:
    setup_no_restart()


# Start iteration loop:
# 1) Calc contacts, hbonds & avelnpi per residue
# 	optional 2) Sample Bc/Bh & obtain ensemble of lambdas
# 2) Update lambdas using previous Bc/Bh. Check for convergence -> break
# 	optional 4) Optimize Bc/Bh with either gradient or MC optimization
# 5) Print parameter values, lambdas & agreement with expt at this step
# 6) Goto 1 
for ncount in range(currcount+1,count+1):
    _contacts = Bc*contacts
    _hbonds = Bh*hbonds
    lnpi=_hbonds+_contacts
    if not mc_sample_params:
      biasfactor = np.sum(lambdas[:, np.newaxis] * lnpi, axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
    else:
      biasfactor = np.sum(lambdasc[:, np.newaxis] * contacts + lambdash[:, np.newaxis] * hbonds , axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
    weights=iniweights*np.exp(biasfactor)
    weights=weights/np.sum(weights)
    avelnpi=np.sum(weights * lnpi, axis=1)


    # On first iteration, set std. dev. of ln(Pf)
    if ncount==1:
       sigmalnpi=np.sum(weights*(lnpi**2), axis=1)
       sigmalnpi=sigmalnpi-(avelnpi**2)
       sigmalnpi=np.sqrt(sigmalnpi)
       avesigmalnpi=np.mean(sigmalnpi)

    avelnpi = np.repeat(avelnpi[:,np.newaxis], len(times), axis=1)
    avelnpi = avelnpi[np.newaxis,:,:].repeat(numseg, axis=0)


    # Do arithmetic always multiplying values by filter, so 'False' entries are not counted
    # Set temp arrays for numerator/denominator so we can use np.divide to avoid divide-by-zero warning
    denom = avelnpi*segfilters
    byres_deutfrac = 1.0 - np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))

    byseg_deutfrac = np.nanmean(byres_deutfrac, axis=1)
    byseg_deutfrac = byseg_deutfrac[:,np.newaxis,:].repeat(segfilters.shape[1], axis=1)
    chisquare=np.sum((byseg_deutfrac*segfilters - exp_df_segf)**2) / normseg 

    # Save initial segfracs
    if ncount == 1:
        if mc_sample_params:
          byres_deutfracav=byres_deutfrac
        np.savetxt("%sinitial_segment_fractions.dat" % prefix, \
                   np.nanmean(byres_deutfrac, axis=1), \
                   header="Times: "+" ".join(str(t) for t in times), \
                   fmt = "%8.5f")
        with open("%sper_iteration_output.dat" % prefix, 'a') as f:
            f.write("%5d %s %s %s %s %s %s %s # INITIAL chisquare before reweighting\n" % (0,0,chisquare,0,0,ratef,Bh,Bc))     
    elif ncount == currcount+1: # If we've restarted
        with open("%sper_iteration_output.dat" % prefix, 'a') as f:
            f.write("%5d %s %s %s %s %s %s %s # RESTARTED chisquare for continuing reweighting\n" % (currcount,0,chisquare,lambdamod,deltalambdamod,ratef,Bh,Bc))     

    # Optionally update Bc/Bh using average values from MC sampling of Bc/Bh & their agreement with expt
    if mc_sample_params:
       if nequil>0:
          parrate=nequil/np.sqrt(nequil*(nequil+ncount))
       else: 
          parrate=1.0
       avecontacts=np.sum(weights*contacts, axis=1)
       avehbonds=np.sum(weights*hbonds, axis=1)
       chisquareavold=chisquareav
       byres_deutfracold=byres_deutfracav
       _lambdanewoldh=_lambdanewh
       _lambdanewoldc=_lambdanewc
       Bhaver=0
       Bcaver=0       
       _lambdanewaverh=0
       _lambdanewaverc=0
       chisquareaver=0
       byres_deutfracsum=0*byres_deutfrac
       Bhold=Bh
       Bcold=Bc 
       for iterpar in range(0,paramcount):

           Bhnew=Bh+((np.random.random_sample())-0.5)*paramsfact*Bhrange
           Bcnew=Bc+((np.random.random_sample())-0.5)*paramsfact*Bcrange
           # recalculate avelnpi with the new parameters

           avelnpinew = Bcnew*avecontacts+Bhnew*avehbonds

           avelnpinew = np.repeat(avelnpinew[:,np.newaxis], len(times), axis=1)
           avelnpinew = avelnpinew[np.newaxis,:,:].repeat(numseg, axis=0)

           # recalculate the simulated deuterated fraction with the new avelnpi

           denomnew = avelnpinew*segfilters
           byres_deutfracnew = 1.0 - np.exp(np.divide(num, np.exp(denomnew), out=np.full(num.shape, np.nan), where=denomnew!=0))

           byseg_deutfracnew = np.nanmean(byres_deutfracnew, axis=1)
           byseg_deutfracnew = byseg_deutfracnew[:,np.newaxis,:].repeat(segfilters.shape[1], axis=1)
           chisquarenew=np.sum((byseg_deutfracnew*segfilters - exp_df_segf)**2) / normseg 
           like=numseg*len(times)*chisquarenew/(2.0*chisquareref)
           likeold=numseg*len(times)*chisquare/(2.0*chisquareref)
           if Bhnew>0:
              if Bcnew>0:
                 if chisquarenew<chisquare:
                    Bh=Bhnew
                    Bc=Bcnew
                    chisquare=chisquarenew
                    avelnpi=avelnpinew
                    denom=denomnew
                    byseg_deutfrac=byseg_deutfracnew
                    byres_deutfrac=byres_deutfracnew 
                 else:
                    a_mov=np.exp(-(like-likeold))
                    if a_mov > np.random.random_sample():
                       Bh=Bhnew
                       Bc=Bcnew
                       chisquare=chisquarenew
                       avelnpi=avelnpinew
                       denom=denomnew
                       byseg_deutfrac=byseg_deutfracnew 
                       byres_deutfrac=byres_deutfracnew
           #print("Parameters:",iterpar,Bh,Bc,chisquare,chisquareref)
           Bhaver=Bhaver+Bh
           Bcaver=Bcaver+Bc
           chisquareaver=chisquareaver+chisquare
           byres_deutfracsum=byres_deutfracsum+byres_deutfrac
           if update_lambdas:
              _lambdanew=np.nansum(np.sum((byseg_deutfrac*segfilters-exp_df_segf) *\
                       np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))*\
                       np.divide(-num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0), axis=2) / \
                       (np.sum(segfilters, axis=1)[:,0])[:,np.newaxis], axis=0)
              _lambdanewaverh=_lambdanewaverh+Bh*_lambdanew        
              _lambdanewaverc=_lambdanewaverc+Bc*_lambdanew
       Bh=Bhold*(1.0-parrate)+parrate*(Bhaver/paramcount)
       Bc=Bcold*(1.0-parrate)+parrate*(Bcaver/paramcount)
       _lambdanewh=_lambdanewoldh*(1.0-parrate)+parrate*(_lambdanewaverh/paramcount)
       _lambdanewc=_lambdanewoldc*(1.0-parrate)+parrate*(_lambdanewaverc/paramcount)
       _lambdanew=0.5*((_lambdanewh/Bh)+(_lambdanewc/Bc))
       chisquareav=chisquareavold*(1.0-parrate)+parrate*(chisquareaver/paramcount)
       byres_deutfracav=byres_deutfracold*(1.0-parrate)+parrate*(byres_deutfracsum/paramcount) 

       if update_lambdas:
         lambdanewh = gamma * _lambdanewh
         lambdanewc = gamma * _lambdanewc         
         lambdanew=0.5*((lambdanewh/Bh)+(lambdanewc/Bc))  
         avehdxdev = np.sum(np.abs(_lambdanew)) / np.sum(lambdanew!=0)
         rate=ratef/(gamma*avehdxdev*avesigmalnpi) # let's think...
         lambdash=lambdash*(1.0-rate)+rate*lambdanewh
         lambdasc=lambdasc*(1.0-rate)+rate*lambdanewc
         lambdas=lambdas*(1.0-rate)+rate*lambdanew
          
    # Calculate new lambda values at this step
    if update_lambdas:

       #lambdasold=lambdas # not used anymore

       if not mc_sample_params: # MC sampling above has already calculated lambdas at this stage
          _lambdanew=np.nansum(np.sum((byseg_deutfrac*segfilters-exp_df_segf) *\
                   np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))*\
                   np.divide(-num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0), axis=2) / \
                   (np.sum(segfilters, axis=1)[:,0])[:,np.newaxis], axis=0)
          lambdanew = gamma * _lambdanew
          avehdxdev = np.sum(np.abs(_lambdanew)) / np.sum(lambdanew!=0)
          rate=ratef/(gamma*avehdxdev*avesigmalnpi) # let's think...
      
          lambdas=lambdas*(1.0-rate)+rate*lambdanew

       # useful to define stop criteria
       lambdamodold=lambdamod   
       lambdamod=np.sum(np.abs(lambdas)) 
       #deltalambdasum=np.sum(sigmalnpi*np.abs(lambdas-lambdasold)) # multiply by sigmalnpi = same scale as lambdamod
      
       deltalambdamodold=deltalambdamod 
       deltalambdamod=(lambdamod-lambdamodold)/lambdamod
    
       if ncount>100: # Only start updating the step size (rate) after the first 100 steps
          if deltalambdamod*deltalambdamodold<0:
             ratef=ratef/rfact
                
       if np.abs(deltalambdamod)<tol:
          break 
      
       if lambdamod<tol:
          break

    # break out at 1st iteration
#    break
# Parameters optimization, working with protection factors (instead of
# deuterated fractions) the parameters can be calculated analytically

    if optimize_params:

       avecontacts=np.sum(weights*contacts, axis=1)
       avehbonds=np.sum(weights*hbonds, axis=1)

       if mc_opt:

          for iterpar in range(0,paramcount):

              Bhold = Bh
              Bcold = Bc
              chisquareold = chisquare
              Bh = Bh+((np.random.random_sample())-0.5)*paramsfact*Bhrange
              Bc = Bc+((np.random.random_sample())-0.5)*paramsfact*Bcrange
              # recalculate avelnpi with the new parameters

              avelnpi = Bc*avecontacts+Bh*avehbonds

              avelnpi = np.repeat(avelnpi[:,np.newaxis], len(times), axis=1)
              avelnpi = avelnpi[np.newaxis,:,:].repeat(numseg, axis=0)

              # recalculate the simulated deuterated fraction with the new avelnpi

              denom = avelnpi*segfilters
              byres_deutfrac = 1.0 - np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))

              byseg_deutfrac = np.nanmean(byres_deutfrac, axis=1)
              byseg_deutfrac = byseg_deutfrac[:,np.newaxis,:].repeat(segfilters.shape[1], axis=1)
              chisquare = np.sum((byseg_deutfrac*segfilters - exp_df_segf)**2) / normseg 
              if Bh<0:
                 Bh = Bhold
                 Bc = Bcold
                 chisquare = chisquareold
              if Bc<0:
                 Bh = Bhold
                 Bc = Bcold
                 chisquare = chisquareold
              if chisquare>chisquareold:
                 Bh = Bhold
                 Bc = Bcold
                 chisquare = chisquareold
       else:

          for iterpar in range(0,paramcount):
              # define first derivative of chisquare respect to Bh
              # define derivative of the sim_dfrac respect to parameters
              # we need average number of contatcs and hbonds
              #
              eff_rate=(np.divide(-num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0)) 
              der_dfracBh=np.nanmean(-(np.exp(-eff_rate)) * \
                          eff_rate * avehbonds[np.newaxis,:, np.newaxis], axis=1) # need a sum over residues if is on fragments 
              der_dfracBc=np.nanmean(-(np.exp(-eff_rate)) * \
                          eff_rate * avecontacts[np.newaxis,:, np.newaxis], axis=1)
              secder_dfracBh=np.nanmean(avehbonds[np.newaxis,:, np.newaxis]*avehbonds[np.newaxis,:, np.newaxis]*\
                             (np.exp(-eff_rate))*\
                             (eff_rate-eff_rate*eff_rate), axis=1)
              secder_dfracBc=np.nanmean(avecontacts[np.newaxis,:, np.newaxis]*avecontacts[np.newaxis,:, np.newaxis]*\
                             (np.exp(-eff_rate))*\
                             (eff_rate-eff_rate*eff_rate), axis=1)
              simdeufrag=np.nanmean((np.where(segfilters, byseg_deutfrac, np.nan)-np.where(segfilters, exp_dfrac, np.nan)), axis=1)
              derBh=2.0*np.sum(simdeufrag*der_dfracBh)
              derBc=2.0*np.sum(simdeufrag*der_dfracBc)
              secderBh=2.0*np.sum(der_dfracBh*der_dfracBh+\
                       secder_dfracBh*simdeufrag)
              secderBc=2.0*np.sum(der_dfracBc*der_dfracBc+\
                       secder_dfracBc*simdeufrag)
              # now update the parameters
              deltaBh=np.abs(derBh/np.abs(secderBh))
              deltaBc=np.abs(derBc/np.abs(secderBc))
              if deltaBh/np.abs(Bh) > paramsfact:
                 Bh = np.abs(Bh-paramsfact*derBh/np.abs(secderBh))
              else:
                 Bh = np.abs(Bh-derBh/np.abs(secderBh))
              if deltaBc/np.abs(Bc) > paramsfact:
                 Bc = np.abs(Bc-paramsfact*derBc/np.abs(secderBc))
              else:
                 Bc = np.abs(Bc-derBc/np.abs(secderBc))
          
              # recalculate avelnpi with the new parameters
          
              avelnpi = Bc*avecontacts+Bh*avehbonds
          
              avelnpi = np.repeat(avelnpi[:,np.newaxis], len(times), axis=1)
              avelnpi = avelnpi[np.newaxis,:,:].repeat(numseg, axis=0)
          
          
              # recalculate the simulated deuterated fraction with the new avelnpi
          
              denom = avelnpi*segfilters
              byres_deutfrac = 1.0 - np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))
          
              byseg_deutfrac = np.nanmean(byres_deutfrac, axis=1)
              byseg_deutfrac = byseg_deutfrac[:,np.newaxis,:].repeat(segfilters.shape[1], axis=1)
          
          
#              with open("%sper_iteration_output_3TT1_run1.dat" % prefix, 'a') as f:
#                   f.write("%s %s %s \n" % (iterpar,Bh,Bc)) 
              if deltaBh/np.abs(Bh)<tol:
                 if deltaBc/np.abs(Bc)<tol:
                    break
          if not update_lambdas:
# Update avehdxdev & chisquare with final segfracs
              _lambdanew=np.nansum(np.sum((byseg_deutfrac*segfilters-exp_df_segf) *\
                                   np.exp(np.divide(num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0))*\
                                   np.divide(-num, np.exp(denom), out=np.full(num.shape, np.nan), where=denom!=0), axis=2) / \
                                   (np.sum(segfilters, axis=1)[:,0])[:,np.newaxis], axis=0)
              lambdanew = gamma * _lambdanew
              avehdxdev = np.sum(np.abs(_lambdanew)) / np.sum(lambdanew!=0)
              chisquare=np.sum((byseg_deutfrac*segfilters - exp_df_segf)**2) / normseg 
              if mc_sample_params:
                 chisquare=chisquareaver/paramcount
              with open("%sper_iteration_output.dat" % prefix, 'a') as f:
                   f.write("%s %5d %s %s %s %s %s %s %s \n" % ("Final after paramopt:",iterpar,avehdxdev,chisquare,lambdamod,deltalambdamod,ratef,Bh,Bc))
              break
    if mc_sample_params:
       chisquare=chisquareav
    # Save iteration step intermediate values. Note that:
    # if gradient opt (not MC) is used, the Bh/Bc will be updated for step n+1, but chisquare & lambdas & avehdxdev will not
    # if MC opt (not gradient) is used, the Bh/Bc and chisquare will be updated for step n+1, but lambdas & avehdxdev will not
    # if MC sampling is used, the Bh/Bc and chisquare will be updated for step n+1, but lambdas & avehdxdev will not
    # This is because of the structure of the iteration loop
    with open("%sper_iteration_output.dat" % prefix, 'a') as f:
        f.write("%5d %s %s %s %s %s %s %s \n" % (ncount,avehdxdev,chisquare,lambdamod,deltalambdamod,rate,Bh,Bc))     
        if not update_lambdas:
            break 

    # do a restart every 100 steps
    if (ncount % 100) == 0:
        # Recalculate everything before calculating the work
        tmpbiasfactor = 0
        tmp_contacts = Bc*contacts
        tmp_hbonds = Bh*hbonds
        tmplnpi = tmp_hbonds + tmp_contacts
        if not mc_sample_params:
          tmpbiasfactor = np.sum(lambdas[:, np.newaxis] * tmplnpi, axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
        else:
          tmpbiasfactor = np.sum(lambdasc[:, np.newaxis] * contacts + lambdash[:, np.newaxis] * hbonds , axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
        tmpweights=iniweights*np.exp(tmpbiasfactor) 
        tmpweights=tmpweights/np.sum(tmpweights)
        tmpavelnpi=np.sum(tmpweights * tmplnpi, axis=1)
        tmpavelnpi = np.repeat(tmpavelnpi[:,np.newaxis], len(times), axis=1)
        tmpavelnpi = tmpavelnpi[np.newaxis,:,:].repeat(numseg, axis=0)
        tmpmeanbias=-kT*np.sum(lambdas*tmpavelnpi[0,:,0])
        tmpbiaspot=-kT*np.sum(np.atleast_2d(lambdas).T*tmplnpi, axis=0)
        tmpwork=np.sum(tmpweights*np.exp((tmpbiaspot-tmpmeanbias)/kT))
        tmpwork=kT*np.log(tmpwork)
        with open("%sper_restart_output.dat" % prefix, 'a') as f:
            f.write("%5d %s %s %s %s %s %s %s %s\n" % (ncount,avehdxdev,chisquare,lambdamod,deltalambdamod,rate,Bh,Bc,tmpwork))     
        paramlist = [ contacts, hbonds, kint, exp_dfrac, segfilters, lambdas, _lambdanew, chisquareav, \
                      _lambdanewaverh, _lambdanewaverc, _lambdanewh, _lambdanewc, lambdasc, lambdash, \
                      nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, \
                      Bc, Bh, gamma, ratef, avesigmalnpi, ncount ]
        pickle.dump(paramlist, open("%srestart.pkl" % prefix, 'wb'))
        print("Restart %srestart.pkl created at step %d" % (prefix, ncount) )

# Recalculate everything before calculating the work

biasfactor = 0
_contacts = Bc*contacts
_hbonds = Bh*hbonds
lnpi = _hbonds+_contacts
if not mc_sample_params:
  biasfactor = np.sum(lambdas[:, np.newaxis] * lnpi, axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
else:
  biasfactor = np.sum(lambdasc[:, np.newaxis] * contacts + lambdash[:, np.newaxis] * hbonds , axis=0) # Sum over all residues, = array of len(nframes). lambdas is 1D array br
weights=iniweights*np.exp(biasfactor) # for later weights=iniweights*np.exp(biasfactor-meanbias)
weights=weights/np.sum(weights)
avelnpi=np.sum(weights * lnpi, axis=1)

avelnpi = np.repeat(avelnpi[:,np.newaxis], len(times), axis=1)
avelnpi = avelnpi[np.newaxis,:,:].repeat(numseg, axis=0)

meanbias=-kT*np.sum(lambdas*avelnpi[0,:,0])
biaspot=-kT*np.sum(np.atleast_2d(lambdas).T*lnpi, axis=0)
work=np.sum(weights*np.exp((biaspot-meanbias)/kT))
work=kT*np.log(work)
if mc_sample_params:
   chisquare=chisquareav
   byres_deutfrac=byres_deutfracav 
# Save final segfracs, weights, work & Bh/Bc lambdamod etc. values
np.savetxt("%sfinal_segment_fractions.dat" % prefix, \
           np.nanmean(byres_deutfrac, axis=1), \
           header="Times: "+" ".join(str(t) for t in times), \
           fmt = "%8.5f")
np.savetxt("%sfinal_weights.dat" % prefix, weights)

with open("%swork.dat" % prefix, 'a') as f:
    f.write("%s %s %s \n" % (gamma, chisquare, work))

with open("%sper_iteration_output.dat" % prefix, 'a') as f:
    f.write("%5d %s %s %s %s %s %s %s # FINAL values at convergence or max iterations\n" % (ncount,avehdxdev,chisquare,lambdamod,deltalambdamod,rate,Bh,Bc))     
    




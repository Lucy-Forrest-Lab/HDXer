#!/usr/bin/env python

# Class for HDX deuterated fraction prediction objects
#
import mdtraj as md
import numpy as np
import os, glob, itertools, pickle
import Functions


class DfPredictor(object):
    """Superclass for all methods that use the general rate equation:

       Df(t) = 1 - exp((kint/Pf)*t)

       making use of an intrinsic rate and protection factor to calculate
       deuteration at a given time T.

       This object contains functions common to all prediction models, such
       as calculation of intrinsic rates from a sequence, but it will NOT
       calculate protection factors or deuterated fractions alone. Initialises
       with a dictionary of default parameters for analysis, accessible as self.params

       Default parameters can either be updated directly in the self.params
       dictionary or by supplying a extra parameters as kwargs during
       initialisation of the child class, e.g.: Radou(cut_nc=1.0) or PH(**param_dict)"""

    def __init__(self, **extra_params):
        """Initialises a deuterated fraction predictor and updates
           parameters associated with it.
           See self.params for default values"""
        # Initialise main parameters with defaults
        self.params = { 'protonly' : True,
                        'save_detailed' : False,
                        'skip_first' : True,
                        'kint_adjs' : None,
                        'kint_params' : None,
                        'times' : [ 0.167, 1.0, 10.0, 120.0],
                        'segfile' : "segfile.txt",
                        'expfile' : None,
                        'logfile' : "HDX_analysis.log",
                        'outprefix' : '' }
        self.params.update(extra_params) # Update main parameter set from kwargs

        # Default rate adjustments for adjacent amino acids
        # Each key is a residue name, each value is [ lgAL, lgAR, lgBL, lgBR ]
        # Values from Nguyen et al., J. Am. Soc. Mass Spec., 2018, 29, 1936-1939

        # Note that these are the 2018 parameters used on Englander's website
        rate_adjs = { 'ALA': [ 0.00, 0.00, 0.00, 0.00 ],
                      'ARG': [ -0.59, -0.32, 0.08, 0.22 ],
                      'ASN': [ -0.58, -0.13, 0.49, 0.32 ],
                      'ASP': [ 0.90, 0.58, 0.10, -0.18 ],
                      'ASH': [ -0.90, -0.12, 0.69, 0.60 ], # Protonated ASP        
                      'CYS': [ -0.54, -0.46, 0.62, 0.55 ],
                      'CYS2': [ -0.74, -0.58, 0.55, 0.46 ], # Disulfide         
                      'GLY': [ -0.22, 0.22, -0.03, 0.17 ],
                      'GLN': [ -0.47, -0.27, 0.06, 0.20 ],
                      'GLU': [ -0.90, 0.31, -0.11, -0.15 ],
                      'GLH': [ -0.60, -0.27, 0.24, 0.39 ], # Protonated GLU        
                      'HIS': [ 0.00, 0.00, -0.10, 0.14 ],  # Acid rates are N/D, 
                                                           # but at pH where His is deprotonated,
                                                           # errors will be negligible       
                      'HIP': [ -0.80, -0.51, 0.80, 0.83 ],
                      'ILE': [ -0.91, -0.59, -0.73, -0.23 ],
                      'LEU': [ -0.57, -0.13, -0.58, -0.21 ],
                      'LYS': [ -0.56, -0.29, -0.04, 0.12 ],
                      'MET': [ -0.64, -0.28, -0.01, 0.11 ],
                      'PHE': [ -0.52, -0.43, -0.24, 0.06 ],
                      'PRO': [ 0.00, -0.19, 0.00, -0.24 ], # Trans PRO        
                      'PROC': [ 0.00, -0.85, 0.00, 0.60 ], # Cis PRO        
                      'SER': [ -0.44, -0.39, 0.37, 0.30 ],
                      'THR': [ -0.79, -0.47, -0.07, 0.20 ],
                      'TRP': [ -0.40, -0.44, -0.41, -0.11 ],
                      'TYR': [ -0.41, -0.37, -0.27, 0.05 ],
                      'VAL': [ -0.74, -0.30, -0.70, -0.14 ],
                      'NT': [ 0.00, -1.32, 0.00, 1.62 ], # N-term NH3+        
                      'CT': [ 0.96, 0.00, -1.80, 0.00 ], # C-term COO-        
                      'CTH': [ 0.05, 0.00, 0.00, 0.00 ], } # C-term COOH

        # Add updates from kint_adjs dictionary if it's given as a kwarg
        if self.params['kint_adjs'] is not None:
            rate_adjs.update(self.params['kint_adjs'])
        self.params['kint_adjs'] = rate_adjs

        # Adjust ordering so value is [ lgAL, lgBL, lgAR, lgBR ] for kint analysis 
        ### THIS IS A DIFFERENT ORDER TO TABLE 2 IN THE REFERENCE ABOVE ###
        _reordered_rate_adjs = { k : v[:] for k, v in self.params['kint_adjs'].items() } # Deep copy
        for i in _reordered_rate_adjs.values():
            i[1], i[2] = i[2], i[1]
        self.params['_reordered_kint_adjs'] = _reordered_rate_adjs
 
        # Default parameters for ka/kb/kw estimations
        # Values from Bai et al., Proteins, 1993, 17, 75-86
        rate_params = { 'lgkAref' : 2.04,
                        'lgkBref' : 10.36,
                        'lgkWref' : -1.5,
                        'EaA' : 14.,
                        'EaB' : 17.,
                        'EaW' : 19.,
                        'R' : 0.001987,
                        'Tref' : 293,
                        'Texp' : 298,
                        'pKD' : 14.87,
                        'pD' : 7.4 }

        # Add updates from kint_adjs dictionary if it's given as a kwarg
        if self.params['kint_params'] is not None:
            rate_params.update(self.params['kint_params'])
        self.params['kint_params'] = rate_params

    def __getstate__(self):
        """Set state of object for pickling.
           Additional attributes can be removed here"""
        odict = self.__dict__.copy()
        delparams = ['t'] # MDTraj trajectory object
        if os.path.exists(self.params['outprefix']+"topology.pkl"):
            delparams.append('top')
        else:
            pickle.dump(odict.pop('top'), open(self.params['outprefix']+"topology.pkl", 'wb'), protocol=-1)

        # Ignore key errors here as deepcopy in Analysis object __add__ also uses
        # this __getstate__
        for k in delparams:
            try:
                del odict[k] 
            except KeyError:
                pass
        return odict

    def __setstate__(self, d):
        """Set state of object after pickling.
           Additional attributes can be added here"""
        self.__dict__ = d
        if os.path.exists(self.params['outprefix']+"topology.pkl"):
            try:
                self.top = pickle.load(open(self.params['outprefix']+"topology.pkl", 'rb'))
            except (IOError, EOFError):
                raise Functions.HDX_Error("Can't read cached topology file %s. "\
                                          "Re-run calculation after removing the file." \
                                           % (self.params['outprefix']+"topology.pkl"))
        else:
            raise Functions.HDX_Error("No such cache file %s. Re-run the calculation, it should be " \
                                      "created automatically if a Df prediction is run." \
                                      % (self.params['outprefix']+"topology.pkl"))

    def pro_omega_indices(self, prolines):
        """Calculates omega dihedrals (CA-C-N-CA) for all proline
           residues in a given prolines array from list_prolines.
    
           Usage: pro_omega_indices(prolines)
           Returns: (atom_indices, w_angles_by_frame)"""

        atom_names = ['CA', 'C', 'N', 'CA']
        offset = np.asarray([-1, -1, 0, 0])

        w_indices = np.zeros((len(prolines),4))
        for i, residx in enumerate(prolines[:,1]):
            curr_offset = offset + residx
            curr_atm_indices = []
            for x in zip(curr_offset, atom_names):
                # Cycle through previous CA/C, current N, CA
                curr_atm_indices.append(self.top.residue(x[0]).atom(x[1]).index)
            w_indices[i] = curr_atm_indices

        return w_indices, md.compute_dihedrals(self.t, w_indices)

    # Assignments for intrinsic rate adjustments:
    # 1) cis/trans prolines
    # 2) disulfides
    # 3) His protonation states
    # 4) N/C termini

    def assign_cis_proline(self):
        """Assigns cis-proline residues on a by-frame basis"""

        prolines = Functions.list_prolines(self.t, log=self.params['logfile'])
        if prolines is None:
            pass
        else:
            outidxs, outangs = self.pro_omega_indices(prolines)
            for i, proidx in enumerate(prolines[:,1]):
                self.top.residue(proidx).cis_byframe = np.logical_and(outangs < np.pi/2, outangs > -1*np.pi/2)[:,i]
                if np.max(self.top.residue(proidx).cis_byframe) > 0:
                    with open(self.params['logfile'], 'a') as f:
                        f.write("Cis-proline found at frame %d for residue %s!\n" % (np.argmax(self.top.residue(proidx).cis_byframe) + 1, self.top.residue(proidx).resSeq))

    def assign_disulfide(self):
        """Assigns residues involved in disulfide bridges"""

        # This assignment is what MDtraj uses for cyx detection 
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return 'SG' in names and 'HG' not in names

        cyx = [res for res in self.top.residues
               if res.name == 'CYS' and isCyx(res)]

        if len(cyx) > 0:
            sg_coords = self.t.xyz[0]
        else: # Catch empty cyx
            with open(self.params['logfile'], 'a') as f:
                f.write("No oxidised cysteines (CYX) found in topology.\n")
            return
        
        self.top.create_standard_bonds()
        self.top._bonds = list(set(self.top._bonds)) # remove duplicates
        self.top.create_disulfide_bonds(sg_coords)

        # Assign disulfides (identical for each frame)
        for b in self.top._bonds:
            if all(i.element.symbol == 'S' for i in b):
                b[0].residue.disulf = np.ones(self.t.n_frames, dtype=bool)
                b[1].residue.disulf = np.ones(self.t.n_frames, dtype=bool)
                with open(self.params['logfile'], 'a') as f:
                    f.write("Disulfide found for residues %s - %s\n" \
                             % (b[0].residue, b[1].residue))

    def assign_his_protonation(self):
        """Assigns protonation state to HIS residues"""

        hisidx = [ r.index for r in self.top.residues if r.code == 'H' ]
        for i in hisidx:
            names = [ a.name for a in self.top.residue(i).atoms ]
            if all(n in names for n in ['HD1','HE2']): # Atom names for doubly protonated His (Hip)
                self.top.residue(i).HIP = np.ones(self.t.n_frames, dtype=bool)
                with open(self.params['logfile'], 'a') as f:
                    f.write("Protonated His assigned for residue %d\n" % self.top.residue(i).resSeq)
#            else:
#                self.top.residue(i).HIP = np.zeros(self.t.n_frames, dtype=bool)

    def assign_termini(self):
        """Assigns flags to N and C terminal residues"""

        nterm_manual, cterm_manual = False, False
        for c in self.top.chains:
            if c.residue(0).is_protein:
                c.residue(0).nterm = np.ones(self.t.n_frames, dtype=bool)
                nterm = c.residue(0)
            else:
                nterm_manual = True
                nterm_idx = self.top.select("protein")[0]
                nterm = self.top.atom(nterm_idx).residue
                nterm.nterm = np.ones(self.t.n_frames, dtype=bool)
            if c.residue(-1).is_protein:
                c.residue(-1).cterm = np.ones(self.t.n_frames, dtype=bool)
                cterm = c.residue(-1)
            else:
                cterm_manual = True
                cterm_idx = self.top.select("protein")[-1]
                cterm = self.top.atom(cterm_idx).residue
                cterm.cterm = np.ones(self.t.n_frames, dtype=bool)
            with open(self.params['logfile'], 'a') as f:
                if any((nterm_manual, cterm_manual)):
                    f.write("One or more of the chain termini is not a protein residue.\n"
                            "This could be because you don't have chain info in your topology,\n"
                            "or because ligands/waters/ions are identified as separate chains.\n"
                            "Selecting termini from protein residues instead, check below:\n")
                
                f.write("N-terminus identified at: %s\nC-terminus identified at: %s\n" \
                         % (nterm, cterm))


    # Helper function to turn sequence-specific rate adjustments to intrinsic acid/base/water rates
    def _adj_to_rates(self, rate_adjs):
        """Helper function for kint().
           Calculates intrinsic rates for a given set of rate adjustments
           [ log(AL), log(BL), log(AR), log(BR) ] taken from Bai et al.

           Usage: _adj_to_rates(rate_adjs)
           Returns: intrinsic_rate"""

        # Calc reference rates at experimental temperature
        # / np.log(10) = conversion from ln to log10
        lgkAexp = self.params['kint_params']['lgkAref'] - (self.params['kint_params']['EaA']
                  /  np.log(10) / self.params['kint_params']['R']) * \
                  (1./self.params['kint_params']['Texp'] - 1./self.params['kint_params']['Tref'])
        lgkBexp = self.params['kint_params']['lgkBref'] - (self.params['kint_params']['EaB']
                  /  np.log(10) / self.params['kint_params']['R']) * \
                  (1./self.params['kint_params']['Texp'] - 1./self.params['kint_params']['Tref'])
        lgkWexp = self.params['kint_params']['lgkWref'] - (self.params['kint_params']['EaW']
                  /  np.log(10) / self.params['kint_params']['R']) * \
                  (1./self.params['kint_params']['Texp'] - 1./self.params['kint_params']['Tref'])

        # Calc log(kA||kB||kW)
        lgkA = lgkAexp + rate_adjs[0] + rate_adjs[2] - self.params['kint_params']['pD']
        lgkB = lgkBexp + rate_adjs[1] + rate_adjs[3] - self.params['kint_params']['pKD'] + self.params['kint_params']['pD']
        lgkW = lgkWexp + rate_adjs[1] + rate_adjs[3]

        kint = 10**lgkA + 10**lgkB + 10**lgkW
        #print(lgkAexp, lgkBexp, lgkWexp, 10**lgkA, 10**lgkB, 10**lgkW)
        return kint

# Intrinsic rate calc:
    def kint(self):
        """Function for calculating intrinsic rates of residues
           in a given topology
       
           Intrinsic exchange rates k_int are computed using equations below.
           k_int = k_A + k_B + k_W
           lgk_A = lgk_A,ref + lgA_L + lgA_R - pD
           lgk_B = lgk_B,ref + lgB_L + lgB_R - pOD
                 = lgk_B,ref + lgB_L + lgB_R - pK_D + pOD
           lgk_W = lgk_W,ref + lgB_L + lgB_R

           Default parameters for the above can be modified in the 
           Radou.params['kint_params'] dictionary. Sequence-based
           rate adjustments can be modified in the 'kint_adjs' and
           '_reordered_kint_adjs' dictionaries.

           Usage: kint()
           Returns: array of by-residue intrinsic rates"""

        kints = np.zeros(len(self.reslist))


     # Adjust residue names for: Cis-Pro, HIP, cystine bridges, GLH/ASH
        reslist = self.reslist.copy()
        for c in self.top.chains:
            if c.n_residues > 1: # Chain of length 1 is probably a ligand - ignore!
                firstres = c.residue(0).index
                secres = next(r.index for r in c._residues[1:] if r.name != 'PRO')
                try:
                    insert_idx = reslist.tolist().index(secres)
                    reslist = np.insert(reslist, insert_idx, firstres) # Insert 'prev' residue for first index of each chain, as we need to define these as NT
                except ValueError:
                    pass
        oldnames = {}
        for i in reslist:
            curr = self.top.residue(i)
            try:
                if np.max(self.cis_byframe[i]): # If cis-proline is true for any frame
                    oldnames[i] = curr.name
                    curr.name = 'PROC'
                    continue
            except AttributeError:
                pass
            try:
                if np.max(curr.HIP): # If His+ is true for any frame
                    oldnames[i] = curr.name
                    curr.name = 'HIP'
                    continue
            except AttributeError:
                pass
            try:
                if np.max(curr.disulf): # If Cys-Cys is true for any frame
                    oldnames[i] = curr.name
                    curr.name = 'CYS2'
                    continue
            except AttributeError:
                pass
            if curr.name == 'GLU': # If Glu has a protonated carboxylate
                try:
                    curr.atom('HE2')
                    oldnames[i] = curr.name
                    curr.name = 'GLH'
                    continue
                except KeyError:
                     pass
            if curr.name == 'ASP': # If Asp has a protonated carboxylate
                try:
                    curr.atom('HD2')
                    oldnames[i] = curr.name
                    curr.name = 'ASH'
                    continue
                except KeyError:
                    pass
            # Assign N/C termini
            try:
                if np.max(curr.nterm): # If nterm is true for any frame
                    oldnames[i] = curr.name
                    curr.name = 'NT'
                    continue
            except AttributeError:
                pass
            try:
                if np.max(curr.cterm): # If cterm is true for any frame
                    oldnames[i] = curr.name
                    try:
                        if (curr.atom('O').n_bonds > 1 or curr.atom('OXT').n_bonds > 1):
                            curr.name = 'CTH'
                            with open(self.params['logfile'], 'a') as f:
                                f.write("It looks like you have a neutral C-terminus (COOH) at residue %s?\n" % curr)
                        else:
                            curr.name = 'CT'
                        continue
                    except KeyError:
                        with open(self.params['logfile'], 'a') as f:
                            f.write("Residue %s is defined as a C-terminus but has no atom O or OXT, is this correct?\n" % curr)
                        curr.name = 'CT'
                        continue
            except AttributeError:
                pass

        for c in self.top.chains:
            firstres = c.residue(0).index
            try:
                del_idx = reslist.tolist().index(firstres)
                reslist = np.delete(reslist, del_idx) # Remove 'prev' residue we inserted above
            except (ValueError, IndexError): # Not in list or no index in array
                pass
        
        try:
            if np.array_equal(reslist, self.reslist):
                pass
            else:
                raise Functions.HDX_Error("Your residue lists for protection factors and intrinsic rates are different. Check your inputs!")
        except AttributeError:
            print("Please generate protection factors before running intrinsic rate calculations.")
            return
        for i, r in enumerate(reslist):
            curr = self.top.residue(r)
            if r != 0:
                prev = self.top.residue(r-1)
            else:
                prev = curr
                kints[i] = np.inf
                continue
            # check for cispro
            if prev.name == 'PROC':
                with open(self.params['logfile'], 'a') as f:
                    f.write("Performing rate calculation on a by-frame basis for residue %s" % prev)
                adj_cis, adj_trans = self.params['_reordered_kint_adjs'][curr.name][0:2], self.params['_reordered_kint_adjs'][curr.name][0:2]
                adj_cis.extend(self.params['_reordered_kint_adjs']['PROC'][2:4])
                adj_trans.extend(self.params['_reordered_kint_adjs']['PRO'][2:4])
                kint_cis = self._adj_to_rates(adj_cis)
                kint_trans = self._adj_to_rates(adj_trans)
                kint_byframe = np.where(prev.cis_byframe, kint_cis, kint_trans)
                kints[i] = np.mean(kint_byframe) # Overall intrinsic rate is adjusted by mean population of cis-pro
            else:
                curr_adjs = self.params['_reordered_kint_adjs'][curr.name][0:2]
                curr_adjs.extend(self.params['_reordered_kint_adjs'][prev.name][2:4])
                kints[i] = self._adj_to_rates(curr_adjs)

        rids = np.asarray([ self.top.residue(i).resSeq for i in reslist ])
        # Save Kints to separate log file, appending filenames for trajectories read as chunks
        if os.path.exists(self.params['outprefix']+"Intrinsic_rates.dat"):
            filenum = len(glob.glob(self.params['outprefix']+"Intrinsic_rates*"))
            np.savetxt(self.params['outprefix']+"Intrinsic_rates_chunk_%d.dat" % (filenum+1),
                       np.stack((rids, kints), axis=1), fmt=['%7d','%18.8f'],
                       header="ResID  Intrinsic rate / min^-1 ") # Use residue indices internally, print out IDs
        else:    
            np.savetxt(self.params['outprefix']+"Intrinsic_rates.dat", np.stack((rids, kints), axis=1),
                       fmt=['%7d','%18.8f'], header="ResID  Intrinsic rate / min^-1 ") # Use residue indices internally, print out IDs

        for residx, oldname in oldnames.items():
            self.top.residue(residx).name = oldname
        self.oldnames = oldnames
        return kints

    # Deuterated fration by residue
    def dfrac(self, write=True, use_self=True, alternate_pfs=None):
        """Function for calculating by-residue deuterated fractions, for
           a set of Protection factors, intrinsic rates, and exposure times
           previously defined for the current Radou object. Optionally write
           out results.

           Alternatively, if use_self=False, take an alternative set of PFs
           not assigned to the current object and perform the same calculation
           on them.

           Usage: dfrac([write=True, use_self=True, alternate_pfs=None])
           Returns: [n_residues, n_times] 2D numpy array of deuterated fractions"""

        if use_self:
            try:
                curr_pfs = np.stack((np.exp(np.mean(self.lnpf_byframe, axis=1)), np.exp(np.std(self.lnpf_byframe, axis=1, ddof=1))), axis=1)
            except AttributeError:
                curr_pfs = np.stack((np.mean(self.pf_byframe, axis=1), np.std(self.pf_byframe, axis=1, ddof=1)), axis=1) # No lnPF for Persson-Halle etc
                
            if len(set(map(len,[self.reslist, self.pfs, self.rates]))) != 1: # Check that all lengths are the same (set length = 1)
                raise Functions.HDX_Error("Can't calculate deuterated fractions, your residue/protection factor/rates arrays are not the same length.")
        else:
            curr_pfs = alternate_pfs
            if len(set(map(len,[self.reslist, alternate_pfs, self.rates]))) != 1: # Check that all lengths are the same (set length = 1)
                raise Functions.HDX_Error("Can't calculate deuterated fractions, your provided residue/protection factor/rates arrays are not the same length.")

        curr_pfs[:,1][np.isinf(curr_pfs[:,1])] = 0

        try:
            fracs = np.zeros((len(self.reslist), len(self.params['times']), 2))
        except TypeError:
            fracs = np.zeros((len(self.reslist), 1, 2))
        for i2, t in enumerate(self.params['times']):
            def _residue_fraction(pf, k, time=t):
                logf = -k / pf[0] * time
                val = 1 - np.exp(logf)
                err = (pf[1]/pf[0]) * k/pf[0] # sd(bA^-1) = rel_sd(A) * A^-1 * b
                logerr = logf + np.log(err) + np.log(time) # sd(e^Aa) = f * sd(A) * a
                err = np.exp(logerr)          # Avoid underflow 
                return np.asarray([val, err])
            for i1, curr_frac in enumerate(map(_residue_fraction, curr_pfs, self.rates)):
                fracs[i1,i2,:] = curr_frac

        rids = np.asarray([ self.top.residue(i).resSeq for i in self.reslist ])
        rids = np.reshape(rids, (len(self.reslist),1))
        # Write resfracs to separate file, appending filenames for trajectories read as chunks
        outfracs = np.reshape(fracs.flatten(), (fracs.shape[0], fracs.shape[1] * fracs.shape[2])) # Reshape as 'Time Err Time Err...' 
        if write:
            if os.path.exists(self.params['outprefix']+"Residue_fractions.dat"):
                filenum = len(glob.glob(self.params['outprefix']+"Residue_fractions*"))
                np.savetxt(self.params['outprefix']+"Residue_fractions_chunk_%d.dat" % (filenum+1),
                           np.concatenate((rids, outfracs), axis=1),
                           fmt='%7d ' + '%8.5f '*len(self.params['times']*2),
                           header="ResID  Deuterated fraction, Times, std. dev / min : %s" \
                           % ' '.join([ str(t) for t in self.params['times'] ])) # Use residue indices internally, print out IDs
            else:    
                np.savetxt(self.params['outprefix']+"Residue_fractions.dat",
                           np.concatenate((rids, outfracs), axis=1),
                           fmt='%7d ' + '%8.5f '*len(self.params['times']*2),
                           header="ResID  Deuterated fraction, Times / min: %s" \
                           % ' '.join([ str(t) for t in self.params['times'] ])) # Use residue indices internally, print out IDs
        return fracs


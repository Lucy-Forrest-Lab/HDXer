#!/usr/bin/env python

# Analysis/plotting functions for HDX analysis

import Functions, Methods
import numpy as np
import matplotlib.pyplot as plt
import os, glob, copy, itertools, pickle
from scipy.stats import pearsonr as correl
from scipy.stats import sem as stderr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, MaxNLocator
from cycler import cycler


### Define defaults for matplotlib plots
plt.rc('lines', linewidth=1.5, markersize=4)
plt.rc('axes', prop_cycle=(cycler('color', ['k','b','r','orange','c','m','y','g'])), # Color cycle defaults to black
       labelweight='heavy', labelsize=14, titlesize=18) # Default fontsizes for printing
plt.rc('axes.spines', top=False, right=False) # Switch off top/right axes
plt.rc('legend', fontsize=10) # Default fontsizes for printing
plt.rc('xtick', labelsize=12) # Default fontsizes for printing
plt.rc('ytick', labelsize=12) # Default fontsizes for printing
plt.rc('figure', titlesize=22, titleweight='heavy') # Default fontsizes for printing
#plt.rc('text', usetex=True)

### Classes
class Analyze():
    """Class to contain results and analysis methods for HDX predictions"""

    def __init__(self, resobj, top, **extra_params):
        """Initialises Analyze object from a Method object with by-residue results"""
        try:
            self.residxs = resobj.reslist
            self.resobj = resobj
            # Analysis ignores errors so far
            # Cumulative resfracs = 3D-array[chunk, resfrac, time]
            self.resfracs = np.reshape(resobj.resfracs[:,:,0], (1, len(resobj.resfracs), len(resobj.resfracs[0])))
            self.c_resfracs = np.copy(self.resfracs)
            # Byframe PFs = 2D-array[residue, PFs]
            self.pf_byframe = np.copy(resobj.pf_byframe)
            if type(resobj) is Methods.Radou:
                self.lnpf_byframe = np.copy(resobj.lnpf_byframe)
            # Cumulative PFs = 2D-array[chunk, PFs]
            self.pfs = np.reshape(resobj.pfs[:,0], (1, len(resobj.pfs)))
            if type(resobj) is Methods.Radou:
                self.lnpfs = np.reshape(resobj.lnpfs[:,0], (1, len(resobj.lnpfs)))
            self.c_pfs = np.copy(self.pfs)
            if type(resobj) is Methods.Radou:
                self.c_lnpfs = np.copy(self.lnpfs)
            # Cumulative n_frames = 1D-array[n_frames]
            self.n_frames = np.atleast_1d(resobj.n_frames)
            self.c_n_frames = np.copy(self.n_frames)
            # Topology & rates
            self.rates = resobj.rates
            self.top = top
            self.resnums = np.asarray([ self.top.residue(i).resSeq for i in self.residxs ])
        except AttributeError:
            raise Functions.HDX_Error("Error when copying results from prediction to analysis objects - have you made any HDX predictions yet?")
        self.params = resobj.params
        try:
            self.params.update(extra_params)
        except (TypeError, ValueError):
            print("Couldn't load extra parameters for analysis (maybe they weren't provided?).\nUsing previous parameters from %s object." % resobj)
                   
    def __add__(self, other):
        """Add resfracs, pfs and n_frames from a second results object and
           update cumulative sums.

           Usage: __add__(self, other)"""

        if isinstance(other, Analyze):
#            try:
                if not all((np.array_equal(self.residxs, other.residxs), np.array_equal(self.rates, other.rates))):
                    print("Reslist or rates of added Analyze objects differ. Not adding them!")
                    return self
                new = copy.deepcopy(self)
                # Add n_frames
                new.n_frames = np.append(new.n_frames, other.n_frames)
                new.c_n_frames = np.cumsum(new.n_frames)
                
                # Calc running ave of PFs = 2D-array[chunk, PFs]
                new.pf_byframe = np.concatenate((new.pf_byframe, other.pf_byframe), axis=1)
                new.pfs = np.concatenate((new.pfs, other.pfs), axis=0)
                _ = np.copy(new.pfs)
                for frames, curr_pf in zip(new.n_frames, _):
                    curr_pf *= frames
                new.c_pfs = np.cumsum(_, axis=0)
                for tot_frames, tot_pf in zip(new.c_n_frames, new.c_pfs):
                    tot_pf /= tot_frames

                # new.c_lnpfs should be calculated from new.lnpf_byframe
                if type(self.resobj) is Methods.Radou:
                    new.lnpf_byframe = np.concatenate((new.lnpf_byframe, other.lnpf_byframe), axis=1)
                    new.lnpfs = np.concatenate((new.lnpfs, other.lnpfs), axis=0)
                    new.c_lnpfs = np.append(new.c_lnpfs, np.mean(new.lnpf_byframe, axis=1)[np.newaxis,:], axis=0)
            
                # Calc running ave of resfracs = 3D-array[chunk, resfrac, time]
                new.resfracs = np.concatenate((new.resfracs, other.resfracs), axis=0)
                _ = np.zeros(new.resfracs[0].shape)
                # Redo resfrac calculation based on running average of pfs
                # N.B. Due to the exponential this is NOT just an average of the resfrac blocks
                if type(self.resobj) is Methods.Radou:
                    for i2, t in enumerate(new.params['times']):
                        def _residue_fraction_lnpf(lnpf, k, time=t):
                            return 1 - np.exp((-k / np.exp(lnpf)) * time)
                        for i1, curr_frac in enumerate(map(_residue_fraction_lnpf, new.c_lnpfs[-1], new.rates)):
                            _[i1,i2] = curr_frac
                else:
                    for i2, t in enumerate(new.params['times']):
                        def _residue_fraction(pf, k, time=t):
                            return 1 - np.exp((-k / pf) * time)
                        for i1, curr_frac in enumerate(map(_residue_fraction, new.c_pfs[-1], new.rates)):
                            _[i1,i2] = curr_frac
                new.c_resfracs = np.concatenate((new.c_resfracs,
                                                 np.reshape(_, (1, len(new.residxs), len(new.params['times'])))),
                                                 axis=0)

                return new
 
#            except AttributeError:
#                raise Functions.HDX_Error("Error when adding analysis objects - have you made any HDX predictions yet?")
        else:
            return self


    def __getstate__(self):
        """Set state of object for pickling.
           Additional attributes can be removed here"""
        odict = self.__dict__.copy()
        for k1 in ['resobj']: # Results object
            try:
                for k2 in ['top']: # topology
                    del odict[k1].__dict__[k2]
            except KeyError:
                pass
        return odict

    def __setstate__(self, d):
        """Set state of object after pickling.
           Additional attributes can be added here"""
        # This will read in a single topology for the whole analysis.
        # It may have attributes that differ from those in self.top
        # e.g. for cis-prolines. These should be recalculated if needed
        # The pfs/rates/fracs in the results object would be correct though.
        self.__dict__ = d
        if os.path.exists(self.params['outprefix']+"topology.pkl"):
            try:
                self.resobj.top = pickle.load(open(self.params['outprefix']+"topology.pkl", 'rb'))
            except (IOError, EOFError):
                raise Functions.HDX_Error("Can't read cached topology file %s. "\
                                          "Re-run calculation after removing the file." \
                                           % (self.params['outprefix']+"topology.pkl"))
        else:
            self.resobj.top = self.top

    def _windowed_average(self, data, window):
        """Calculate average of non-overlapping windows (size=window) of a set of data.

           Usage: _windowed_average(data, window)"""

        blocks = len(data)/window
        aves = []
        for start_i in range(int(blocks)):        
            aves.append(np.mean(data[(start_i * window):(start_i * window) + window]))
        return np.asarray(aves)


    def _cumulative_average(self, data, blocksizes):
        """Calculate cumulative averages of a set of data at provided intervals
           Data & blocksizes should be 1D arrays (or axis-slices of larger arrays)
           
           Usage: _cumulative_average(data, blocksizes)"""
        if not len(data) == np.sum(blocksizes):
            raise Functions.HDX_Error("Unable to cumulatively average data of length %d using total blocksizes %d" \
                                      % (len(data), int(np.sum(blocksizes))))
        aves = np.zeros(len(blocksizes))
        blocksum = np.cumsum(blocksizes)
        for i, block in enumerate(blocksum):
            aves[i] = np.mean(data[:block])
        return aves


    def read_segfile(self):

        # segfile should contain at most 3 columns: startres, endres, chain_idx
        try:
            self.segres = np.loadtxt(self.params['segfile'],
                                     dtype=[ ('segres', np.int32, (2,)), ('chain', np.int32, (1)) ])  # ResIDs will be converted to indices with dictionary in segments function
            with open(self.params['logfile'], 'a') as f:
                f.write("Chain indices read from segments file - segment averaging will be performed on defined chains\n")
            self._single_chain = False
        except IndexError:
            tmp_segres = np.loadtxt(self.params['segfile'], dtype=np.int32, usecols=(0,1)) 
            with open(self.params['logfile'], 'a') as f:
                f.write("Chain indices NOT read from segments file - segment averaging will be performed on first chain\n")
            self.segres = np.zeros(len(tmp_segres), dtype=[ ('segres', np.int32, (2,)), ('chain', np.int32, (1)) ])
            self.segres['segres'] = tmp_segres            
            self._single_chain = True
        except ValueError:
            raise Functions.HDX_Error("There's a problem reading the values in your segments file: %s \n"
                                      "File should contain either 2 or 3 columns of integers, separated by spaces.\n"
                                      "Format: start_residue end_residue chain_index[optional]")


    def read_expfile(self):
        """Reads an experimental data file for comparison to predicted data.

           Experimental results file should be formatted as follows:
           Seg_start   Seg_end   Time_1   Time_2   Time_3 ... [Time_n]

           This is the same format as the printout of predicted results"""

        # Check I'm not loading in too many timepoints
        try:
            if self._single_chain:
                expt = np.loadtxt(self.params['expfile'], dtype=[ ('segres', np.int32, (2,)),
                                  ('fracs', np.float64, (len(self.params['times']),)) ])
            else:
                expt = np.loadtxt(self.params['expfile'], dtype=[ ('segres', np.int32, (2,)),
                                  ('chain', np.int32, (1)), ('fracs', np.float64, (len(self.params['times']),)) ])
        except ValueError as err:
            raise Functions.HDX_Error("There's a problem with the experimental data file. It has too few timepoints. \n" \
                                      "This can be caused if you've defined chain indices in the segments file but not in the experimental data file.\n" \
                                      "The error while reading was: %s" % str(err))
        # Now check I'm not loading in too few timepoints
        try:
            if self._single_chain:
                expt = np.loadtxt(self.params['expfile'], dtype=[ ('segres', np.int32, (2,)),
                                  ('fracs', np.float64, (len(self.params['times']) + 1,)) ])
            else:
                expt = np.loadtxt(self.params['expfile'], dtype=[ ('segres', np.int32, (2,)),
                                  ('chain', np.int32, (1)), ('fracs', np.float64, (len(self.params['times']) + 1,)) ])
            raise Functions.HDX_Error("There's a problem with the experimental data file. It has too many timepoints. \n" 
                                      "This can be caused if you've defined chain indices in the experimental data file but not in the segments file.\n")
        except ValueError:
            pass
        # Check expt = predicted
        if self._single_chain:
            if np.array_equal(self.segres['segres'], expt['segres']):
                self.expfracs = expt['fracs']
            else:
                raise Functions.HDX_Error("The experimental segments read from %s and predicted segments read from %s don't match!" % (self.params['segfile'], self.params['expfile']))
        else:                               
            if all( (np.array_equal(self.segres['segres'], expt['segres']), np.array_equal(self.segres['chain'], expt['chain'])) ):
                self.expfracs = expt['fracs']
            else:
                raise Functions.HDX_Error("The experimental segments/chains read from %s and predicted segments/chains read from %s don't match!" % (self.params['segfile'], self.params['expfile']))
    
    def segments(self, top):
        """Function to average residue deuterated fractions over
           a given set of peptide segments. The first residue in each
           segment will not be included in the averaging, as it is assumed
           to be 100% back-exchanged during analysis.
    
           Residue indices provided in the given list are converted to 
           residue IDs from the given trajectory's topology. Currently this
           remumbering will only work for single chain topologies with sequential
           numbering. If a residue in a segment is not found (e.g. a truncated
           N/C terminus), the next residue is chosen as the start/end point instead.
 
           Writes info on skipped residues to logfile "HDX_analysis.log" by default
           and the segment/average deuteration information to "Segment_average_fractions.dat"

           Usage: segments(traj, residxs, fracs, segfile_name, times, [ log="HDX_analysis.log" ])
           Returns: [n_segs, 2] 2D numpy array of segment start/end residue IDs, 
                [n_segs, n_times] 2D numpy array of segment deuterated fractions at each timepoint"""

        res2idx = {}
        with open(self.params['logfile'], 'a') as f:
            f.write("Now converting residue numbers to indices for segment averaging:\n")
        for idx, res in enumerate(top.residues):
            if res.is_protein:
                res2idx[(res.resSeq, res.chain.index)] = idx 
#                res2idx[res.resSeq] = idx # Only works for single chain or sequential numbers, no re-use of resnums
            else:
                with open(self.params['logfile'], 'a') as f:
                    f.write("Skipping residue: %s, not a protein residue\n" % res)
        
        
        self.read_segfile()
        try:
            aves = np.zeros((len(self.resfracs), len(self.segres), len(self.params['times'])))
            stddevs = np.zeros((len(self.resfracs), len(self.segres), len(self.params['times'])))
            c_aves = np.zeros((len(self.c_resfracs), len(self.segres), len(self.params['times'])))
        except TypeError:
            aves = np.zeros((len(self.resfracs), len(self.segres), 1))
            stddevs = np.zeros((len(self.resfracs), len(self.segres), 1))
            c_aves = np.zeros((len(self.c_resfracs), len(self.segres), 1))
            self.params['times'] = [self.params['times']]    

        # Info for 'skip_first'
        if self.params['skip_first']:
            for i1, (seg, chain) in enumerate(self.segres):
                with open(self.params['logfile'], 'a') as f:
                    try:
                        f.write("'Skip_first' is set. Not including residue %s in averaging for segment %s-%s, chain idx %s.\n" \
                                % (top.residue(res2idx[(seg[0], chain)]), seg[0], seg[1], chain))
                    except KeyError:
                        _ = top.chain(chain).residue(0)
                        f.write("'Skip_first' is set. Not including residue %s in averaging for segment %s-%s, chain idx %s.\n" \
                                % (_, seg[0], seg[1], chain))
                        
        else:
            for i1, (seg, chain) in enumerate(self.segres):
                with open(self.params['logfile'], 'a') as f:
                    try:
                        f.write("'Skip_first' is NOT set. Including residue %s in averaging for segment %s-%s, chain idx %s.\n" \
                                % (top.residue(res2idx[(seg[0], chain)]), seg[0], seg[1], chain))
                    except KeyError:
                        _ = top.chain(chain).residue(0)
                        f.write("'Skip_first' is NOT set. Including residue %s in averaging for segment %s-%s, chain idx %s.\n" \
                                % (_, seg[0], seg[1], chain))

        # Calc average fractions for each chunk                    
        for i0, chunk, errchunk in zip(range(len(self.resfracs)), self.resfracs, self.resfrac_STDs):
            for i2, t in enumerate(self.params['times']):
                for i1, (seg, chain) in enumerate(self.segres):
                    try:
                        start = res2idx[(seg[0], chain)]
                    except KeyError:
                        with open(self.params['logfile'], 'a') as f:
                            f.write("Didn't find residue %s, chain %s in protein. Using residue %s, chain %s as startpoint instead.\n" \
                                 % (seg[0], chain, top.chain(chain).residue(0), chain))
                        start = top.chain(chain).residue(0).index
                    try:
                        end = res2idx[(seg[1], chain)]
                    except KeyError:
                        with open(self.params['logfile'], 'a') as f:
                            f.write("Didn't find residue %s, chain %s in protein. Using residue %s, chain %s as endpoint instead.\n" \
                                     % (seg[1], chain, top.chain(chain).residue(-1), chain))
                        end = top.chain(chain).residue(-1).index
    
                    if self.params['skip_first']:
                        idxs = np.where(np.logical_and( self.residxs > start, self.residxs <= end ))[0] # > start skips
                    else:
                        idxs = np.where(np.logical_and( self.residxs >= start, self.residxs <= end ))[0] # >= start incs

                    aves[i0, i1, i2] = np.mean(chunk[idxs, i2])
                    stddevs[i0, i1, i2] = np.sqrt(np.sum(errchunk[idxs, i2]**2)) / len(np.nonzero(idxs))

        stderrs = np.copy(stddevs)
        for i0, a in enumerate(stderrs):
            a /= np.sqrt(self.n_frames[i0])

        # Do the same for cumulative resfracs
        for i0, cchunk in enumerate(self.c_resfracs):
            for i2, t in enumerate(self.params['times']):
                for i1, (seg, chain) in enumerate(self.segres):
                    try:
                        start = res2idx[(seg[0], chain)]
                    except KeyError:
                        with open(self.params['logfile'], 'a') as f:
                            f.write("Cumulative segment averages: "
                                    "Didn't find residue %s, chain %s in protein. Using residue %s, chain %s as startpoint instead.\n" \
                                    % (seg[0], chain, top.chain(chain).residue(0), chain))
                        start = top.chain(chain).residue(0).index
                    try:
                        end = res2idx[(seg[1], chain)]
                    except KeyError:
                        with open(self.params['logfile'], 'a') as f:
                            f.write("Cumulative segment averages: "
                                    "Didn't find residue %s, chain %s in protein. Using residue %s, chain %s as endpoint instead.\n" \
                                    % (seg[0], chain, top.chain(chain).residue(-1), chain))
                        end = top.chain(chain).residue(-1).index
    
                    if self.params['skip_first']:
                        idxs = np.where(np.logical_and( self.residxs > start, self.residxs <= end ))[0] # > start skips
                    else:
                        idxs = np.where(np.logical_and( self.residxs >= start, self.residxs <= end ))[0] # >= start incs

                    c_aves[i0, i1, i2] = np.mean(cchunk[idxs, i2])
                    
        # Write average fractions file for each chunk
        # N.B Again, these will NOT add up to the c_segfracs value, which is recalc'd using
        # the exponential decay and the mean PF at a given timepoint (not just a straight ave
        # of the block averaged resfracs)
        if self._single_chain:
            for chunkave in aves:
                if os.path.exists(self.params['outprefix']+"Segment_average_fractions.dat"):
                    filenum = len(glob.glob(self.params['outprefix']+"Segment_average_fractions*"))
                    np.savetxt(self.params['outprefix']+"Segment_average_fractions_chunk_%d.dat" % (filenum+1),
                               np.hstack((self.segres['segres'], chunkave)),
                               fmt='%6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
                else:
                    np.savetxt(self.params['outprefix']+"Segment_average_fractions.dat", np.hstack((self.segres['segres'], chunkave)),
                               fmt='%6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
        else:
            for chunkave in aves:
                if os.path.exists(self.params['outprefix']+"Segment_average_fractions.dat"):
                    filenum = len(glob.glob(self.params['outprefix']+"Segment_average_fractions*"))
                    np.savetxt(self.params['outprefix']+"Segment_average_fractions_chunk_%d.dat" % (filenum+1),
                               np.hstack((self.segres['segres'], self.segres['chain'].reshape((len(self.segres['segres']),1)), chunkave)),
                               fmt='%6d %6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Chain  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
                else:
                    np.savetxt(self.params['outprefix']+"Segment_average_fractions.dat", np.hstack((self.segres['segres'],
                               self.segres['chain'].reshape((len(self.segres['segres']),1)), chunkave)),
                               fmt='%6d %6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Chain  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))

        with open(self.params['logfile'], 'a') as f:
            f.write("Segment averaging complete.\n")

        return aves, c_aves, stddevs, stderrs


    def check_blocksize(self):
        """Evaluate convergence of standard error in the mean for PFs.

           By-frame PFs are successively block averaged at every possible
           block size (1 -> n_frames-1), the SEM calculated across block
           averages, and saved to self.tot_SEMs"""

#        self.tot_SEMs = np.zeros((len(self.pf_byframe)-1, 2), dtype=[np.int32, np.float64])
#       Array(window, SEM)
        valid_windows = []
        for window in range(1, int((self.c_n_frames[-1] / 2)) + 1):
            if self.c_n_frames[-1] % window == 0:
                valid_windows.append(window)
        with open(self.params['logfile'], 'a') as f:
            f.write("Total frames divisible by: %s,\nEvaluating standard error in total PF at these windows.\n"\
                    % " ".join([ str(i) for i in valid_windows ]))
        valid_windows = np.asarray(valid_windows, dtype=np.int32)
        if len(valid_windows) > 0: # 1 or prime frames
            self.tot_SEMs = np.zeros((len(valid_windows), 2)) 
        else:
            self.tot_SEMs = np.zeros((1, 2)) 
        with np.errstate(invalid='ignore'): # Ignores infs in stderr calc
            for i, window in enumerate(valid_windows):
                self.tot_SEMs[i, 0] = window
                self.tot_SEMs[i, 1] = stderr(self._windowed_average(np.sum(self.pf_byframe, axis=0), window))
            self.norm_tot_SEMs = np.copy(self.tot_SEMs) 
            self.norm_tot_SEMs[:,1] /= np.max(self.tot_SEMs[:,1]) # Normalised to max

#       Array(res, window, SEM)
        if len(valid_windows) > 0: # 1 or prime frames
            self.res_SEMs = np.zeros((len(self.resnums), len(valid_windows), 2))
            self.res_STDs = np.zeros((len(self.resnums), len(valid_windows), 2))
        else:
            self.res_SEMs = np.zeros((len(self.resnums), 1, 2))
            self.res_STDs = np.zeros((len(self.resnums), 1, 2))
        with np.errstate(invalid='ignore'): # Ignores infs in stderr calc
            for j, res in enumerate(self.resnums):
                for i, window in enumerate(valid_windows):
                    self.res_SEMs[j,i,0] = window
                    self.res_SEMs[j,i,1] = stderr(self._windowed_average(self.pf_byframe[j], window))
                    self.res_STDs[j,i,1] = np.std(self._windowed_average(self.pf_byframe[j], window), ddof=1)
            self.norm_res_SEMs = np.copy(self.res_SEMs)
            for res in self.norm_res_SEMs:
                res[:,1] /= np.max(res[:,1]) # Normalised to max


    def propagate_errors(self):
        """Propagate errors for individual blocks. Save as std errors for PFs, resfracs & segfracs"""

        self.pf_stds = np.zeros(self.pfs.shape)
        self.pf_SEMs = np.zeros(self.pfs.shape)
        startframe = 0        
        for i, endframe in enumerate(self.c_n_frames):
            self.pf_stds[i] = np.std(self.pf_byframe[:,startframe:endframe], axis=1, ddof=1)
            self.pf_SEMs[i] = stderr(self.pf_byframe[:,startframe:endframe], axis=1)
            startframe += self.n_frames[i]
        
        self.resfrac_STDs = np.zeros(self.resfracs.shape)
        self.resfrac_SEMs = np.zeros(self.resfracs.shape)
        for i in range(self.pfs.shape[0]):
            self.resfrac_STDs[i] = self.resobj.dfrac(write=False, use_self=False, alternate_pfs=np.stack((self.pfs[i], self.pf_stds[i]), axis=1))[:,:,1]
            denom = np.sqrt(self.n_frames[i])
            self.resfrac_SEMs[i] = self.resfrac_STDs[i] / denom  


    def desc_stats(self):
        """Calculates descriptive statistics of segments compared to expt
           for all timepoints"""

        self.correls = np.zeros(len(self.params['times']))
        self.MUE = np.zeros(len(self.params['times']))
        self.MSE = np.zeros(len(self.params['times']))
        for idx, t in enumerate(self.params['times']):
            self.correls[idx] = correl(self.c_segfracs[-1][:,idx], self.expfracs[:,idx])[0]
            self.MSE[idx] = np.mean(self.c_segfracs[-1][:,idx] - self.expfracs[:,idx])
            self.MUE[idx] = np.mean(np.abs(self.c_segfracs[-1][:,idx] - self.expfracs[:,idx]))
        with open(self.params['outprefix']+"Descriptive_statistics.dat", 'a') as f:
            np.savetxt(f, self.correls, header="Pearson's R correlation, Times / min: %s" \
                       % ' '.join([ str(t) for t in self.params['times'] ]), fmt='%8.6f')
            np.savetxt(f, self.MSE, header="Mean signed error / frac, Pred. - Expt., Times / min: %s" \
                       % ' '.join([ str(t) for t in self.params['times'] ]), fmt='%10.8f')
            np.savetxt(f, self.MUE, header="Mean unsigned error / frac, Pred. - Expt., Times / min: %s" \
                       % ' '.join([ str(t) for t in self.params['times'] ]), fmt='%10.8f')

    def print_summaries(self):
        """Print summary PF, resfrac and segment results - for example of a method
           object that has had results summed over multiple chunks."""

        
        # Save PFs to 'SUMMARY' file
        try:
            if os.path.exists(self.params['outprefix']+"SUMMARY_protection_factors.dat"):
                filenum = len(glob.glob(self.params['outprefix']+"SUMMARY_protection_factors*"))
                np.savetxt(self.params['outprefix']+"SUMMARY_protection_factors_%d.dat" % (filenum+1),
                           np.stack((self.resnums, self.c_pfs[-1]), axis=1), fmt=['%7d','%18.8f'],
                           header="ResID  Protection factor") # Use residue indices internally, print out IDs
            else:    
                np.savetxt(self.params['outprefix']+"SUMMARY_protection_factors.dat", np.stack((self.resnums, self.c_pfs[-1]), axis=1),
                           fmt=['%7d','%18.8f'], header="ResID  Protection factor") # Use residue indices internally, print out IDs
        except AttributeError:
            raise Functions.HDX_Error("Can't write summary protection factors - perhaps you haven't calculated them yet?")

        # Save log(PFs) to 'SUMMARY' file
        try:
            if os.path.exists(self.params['outprefix']+"SUMMARY_logProtection_factors.dat"):
                filenum = len(glob.glob(self.params['outprefix']+"SUMMARY_logProtection_factors*"))
                np.savetxt(self.params['outprefix']+"SUMMARY_logProtection_factors_%d.dat" % (filenum+1),
                           np.stack((self.resnums, self.c_lnpfs[-1]), axis=1), fmt=['%7d','%18.8f'],
                           header="ResID  ln(Protection factor)") # Use residue indices internally, print out IDs
            else:    
                np.savetxt(self.params['outprefix']+"SUMMARY_logProtection_factors.dat", np.stack((self.resnums, self.c_lnpfs[-1]), axis=1),
                           fmt=['%7d','%18.8f'], header="ResID  ln(Protection factor)") # Use residue indices internally, print out IDs
        except AttributeError:
            if type(self.resobj) is Methods.Radou:
                raise Functions.HDX_Error("Can't write summary log protection factors - perhaps you haven't calculated them yet?")
            else:
                pass

        # Save residue deuterated fractions to 'SUMMARY' file
        try:
            if os.path.exists(self.params['outprefix']+"SUMMARY_residue_fractions.dat"):
                filenum = len(glob.glob(self.params['outprefix']+"SUMMARY_residue_fractions*"))
                np.savetxt(self.params['outprefix']+"SUMMARY_residue_fractions_%d.dat" % (filenum+1),
                           np.concatenate((np.reshape(self.resnums, (len(self.residxs),1)), self.c_resfracs[-1]), axis=1),
                           fmt='%7d ' + '%8.5f '*len(self.params['times']),
                           header="ResID  Deuterated fraction, Times / min: %s" \
                           % ' '.join([ str(t) for t in self.params['times'] ])) # Use residue indices internally, print out IDs
            else:    
                np.savetxt(self.params['outprefix']+"SUMMARY_residue_fractions.dat",
                           np.concatenate((np.reshape(self.resnums, (len(self.residxs),1)), self.c_resfracs[-1]), axis=1),
                           fmt='%7d ' + '%8.5f '*len(self.params['times']),
                           header="ResID  Deuterated fraction, Times / min: %s" \
                           % ' '.join([ str(t) for t in self.params['times'] ])) # Use residue indices internally, print out IDs
        except AttributeError:
            raise Functions.HDX_Error("Can't write summary residue fractions - perhaps you haven't calculated them yet?")


        # Save segment deuterated averages to 'SUMMARY' file
        try:
            if self._single_chain: 
                if os.path.exists(self.params['outprefix']+"SUMMARY_segment_average_fractions.dat"):
                    filenum = len(glob.glob(self.params['outprefix']+"SUMMARY_segment_average_fractions*"))
                    np.savetxt(self.params['outprefix']+"SUMMARY_segment_average_fractions_%d.dat" % (filenum+1),
                               np.hstack((self.segres['segres'], self.c_segfracs[-1])),
                               fmt='%6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
                else:
                    np.savetxt(self.params['outprefix']+"SUMMARY_segment_average_fractions.dat",
                               np.hstack((self.segres['segres'], self.c_segfracs[-1])),
                               fmt='%6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
            else:
                if os.path.exists(self.params['outprefix']+"SUMMARY_segment_average_fractions.dat"):
                    filenum = len(glob.glob(self.params['outprefix']+"SUMMARY_segment_average_fractions*"))
                    np.savetxt(self.params['outprefix']+"SUMMARY_segment_average_fractions_%d.dat" % (filenum+1),
                               np.hstack((self.segres['segres'], self.segres['chain'].reshape((len(self.segres['segres']),1)), self.c_segfracs[-1])),
                               fmt='%6d %6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Chain  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
                else:
                    np.savetxt(self.params['outprefix']+"SUMMARY_segment_average_fractions.dat",
                               np.hstack((self.segres['segres'], self.segres['chain'].reshape((len(self.segres['segres']),1)), self.c_segfracs[-1])),
                               fmt='%6d %6d %6d ' + '%8.5f '*len(self.params['times']), header="Res1   Res2  Chain  Times / min: %s" \
                               % ' '.join([ str(t) for t in self.params['times'] ]))
                

        except AttributeError:
            raise Functions.HDX_Error("Can't write summary segment fractions - perhaps they haven't been calculated yet?")
        
    
    @Functions.cacheobj()    
    def run(self):
        """Run a by-segment HDX prediction and optionally compares to experiment"""

        self.pf_byframe = np.nan_to_num(self.pf_byframe)
        if type(self.resobj) is Methods.Radou:
            self.lnpf_byframe = np.nan_to_num(self.lnpf_byframe)
#        self.resfracs = self.resobj.dfrac(write=True, use_self=False, alternate_pfs=np.stack((np.exp(np.mean(self.lnpf_byframe, axis=1)), np.exp(np.std(self.lnpf_byframe, axis=1, ddof=1))), axis=1))
        self.read_segfile()
        self.check_blocksize()
        self.propagate_errors()
        self.segfracs, self.c_segfracs, self.seg_STDs, self.seg_SEMs = self.segments(self.top)
        if self.params['expfile'] is not None:
            self.read_expfile()
            self.desc_stats()
        return self # For consistency with pickle


### Analysis object for multiple runs (Not yet implemented!)
class MultiAnalyze():
    """Class to contain results and analysis methods for HDX predictions"""

    def __init__(self, filenames):
        """Initialise a MultiAnalyze object from multiple individual
           Analyze objects. All results are taken as the arithmetic mean
           across the objects, with uncertainties taken as the corresponding
           std. err. / std. dev."""

        objects = [ pickle.load(open(fn, 'rb')) for fn in filenames ] 
        summed_objs = sum(objects[1:], objects[0]) # Checks for object type & equivalence of rates/residxs
        to_average = [ '' ]


### Plotting Class
class Plots():
    """Class to plot results of HDX predictions"""

    def __init__(self, aobj):
        """Initialise a Plots object from an Analyze object"""
        if isinstance(aobj, Analyze):
            self.results = aobj
        else:
            raise Functions.HDX_Error("Can't initialize a Plots object from anything"\
                                      "other than a completed Analyze object.")

    def choose_plots(self, **override_opts):
        """Choose available plots based on results in Analyze object.
           Normally these will be automatically chosen based on available
           data, but switch can be overriden by providing kwargs.

           Available plots:
           df_curves : By-segment deuterated fractions for all timepoints
           df_convergence : Convergence of by-segment deuterated fractions across all simulation chunks
           seg_curve : By-segment predictions across all timepoints
           pf_byres : By-residue protection factors
           tot_pf   : Convergence of total protection factor across all simulation chunks
           pf_error : Convergence of standard errors in total protection factor, with respect to block size.
           switch_func : Shape of switching function if a switching function is used
           _expt_overlay : Option to switch on/off overlay of experimental values on all relevant plots
           _block_ave : Option to switch on/off block averaging plots as well as convergence (running ave) (currently unused)

           Sets Plots.avail attribute with dictionary of results."""

        self.avail = { 'df_curve' : False,
                       'df_convergence' : False,
                       'seg_curve' : False,
                       'pf_byres' : False,
                       'tot_pf' : False,
                       'pf_error' : False,
                       'switch_func' : False,
                       '_expt_overlay' : False,
                       '_block_ave' : False } # Currently unused

        self._funcdict = { 'df_curve' : self.df_curve,
                           'df_convergence' : self.df_convergence,
                           'seg_curve' : self.seg_curve,
                           'pf_byres' : self.pf_byres,
                           'tot_pf' : self.tot_pf,
                           'pf_error' : self.pf_error, 
                           'switch_func' : self.switch_func }

        if self.results.resobj.params['contact_method'] == 'switch': # No need for try/except as contact_method should always exist
            self.avail['switch_func'] = True
        try:
            self.results.c_resfracs[-1]
            self.avail['df_curve'] = True
            if len(self.results.c_resfracs) > 1:
                self.avail['df_convergence'] = True
        except (AttributeError, IndexError):
            pass
        try:
            self.results.c_segfracs[-1]
            self.avail['seg_curve'] = True
        except (AttributeError, IndexError):
            pass
        try:
            self.avail['pf_byres'] = ( len(self.results.c_pfs[-1]) == len(self.results.residxs) )
            if len(self.results.c_pfs) > 1:
                self.avail['tot_pf'] = True
            if len(self.results.tot_SEMs) > 1:
                self.avail['pf_error'] = True
        except (AttributeError, IndexError):
            pass
        try:
            # Other data soundness checks (for times, segres) are in Analyze.read_expfile
            self.avail['_expt_overlay'] = ( len(self.results.c_segfracs[-1]) == len(self.results.expfracs) )
        except (AttributeError, IndexError):
            pass

        # Overrides
        if override_opts == None:
            with open(self.results.params['logfile'], 'a') as f:
                f.write("Available plots automatically chosen without overrides\n")
        else:
            try:
                self.avail.update(override_opts)
                with open(self.results.params['logfile'], 'a') as f:
                    f.write("Available plots manually overriden for plots: %s \n" % ", ".join(override_opts.keys()))
            except (TypeError, ValueError):
                with open(self.results.params['logfile'], 'a') as f:
                    f.write("Available plots automatically chosen without overrides\n")


    def _fix_ticks(self, ticklist, maxdata, interval, mindata=0):
        """Fix list of tick positions to finish on maxdata and skip
           penultimate tick if it's closer than interval/2 to maxdata"""

        # Prepend with min        
        ticklist = list(filter(lambda x: x >= mindata, ticklist))
        if ticklist[0] == mindata:
            pass
        else:
            ticklist = np.append(np.asarray([mindata]), ticklist)
        

        # Append with max
        ticklist = list(filter(lambda y: y <= maxdata, ticklist))
        if ticklist[-1] == maxdata:
            return ticklist
        elif maxdata - ticklist[-1] <= interval/2:
            ticklist[-1] = maxdata
            return ticklist
        else:
            ticklist = np.append(ticklist, maxdata)
            return ticklist

        
    def df_curve(self):
        """Plot a predicted deuteration curve for each segment. Plots are optionally
           overlaid with experimental curves, according to the value of
           Plots.avail['_expt_overlay'].

           Plots are saved to a multi-page PDF file df_curves.pdf, with up to
           8 segments per page.""" 

        def _plot_df_curve(ax, segdata, blockdata, overlay_ys=None, **plot_opts):
            xs, ys = self.results.params['times'], segdata[2:]
            ax.plot(xs, ys, marker='o', color='black', linewidth=1.5, linestyle='-', label="Predicted", **plot_opts)
            with np.errstate(all='raise'): # frames length 1 for PH
                try:
                    segmaxs = ys + np.std(blockdata[:,0], axis=0, ddof=1)
                    segmins = ys - np.std(blockdata[:,0], axis=0, ddof=1)
                except (FloatingPointError, RuntimeWarning):
                    segmaxs = ys + np.std(blockdata[:,0], axis=0) # 0.0 std dev
                    segmins = ys - np.std(blockdata[:,0], axis=0) # 0.0 std dev
            ax.fill_between(xs, segmaxs, segmins, color='gray', alpha=0.3, label="Std. Dev. across trajectory blocks")
            ax.set_title("Segment %d-%d" % (segdata[0], segdata[1]), fontsize=9)
            ax.set_xlim(0.0 - xs[-1]*0.05, xs[-1] *1.05) # 5% buffers - do this as log?
            ax.xaxis.set_major_locator(MultipleLocator(30))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 30)
            ax.set_xticks(xticknums)
            ax.set_yticks(np.arange(0.0, 1.2, 0.2))
            ax.set_xlim(0.0 - xs[-1]*0.05, xs[-1] *1.05) # 5% buffers - do this as log?
            ax.set_ylim(-0.05, 1.05)

            if overlay_ys is not None:
                ax.plot(xs, overlay_ys, marker='^', color='blue', linewidth=1.5, linestyle=':', label="Experimental")
            ax.legend(fontsize=6)

        def _plot_log_df_curve(ax, segdata, blockdata, overlay_ys=None, **plot_opts):
            xs, ys = self.results.params['times'], segdata[2:]
            ax.plot(xs, ys, marker='o', color='black', linewidth=1.5, linestyle='-', label="Predicted", **plot_opts)
            with np.errstate(all='raise'): # length 1 arrs for PH
                try:
                    segmaxs = ys + np.std(blockdata[:,0], axis=0, ddof=1)
                    segmins = ys - np.std(blockdata[:,0], axis=0, ddof=1)
                except (FloatingPointError, RuntimeWarning):
                    segmaxs = ys + np.std(blockdata[:,0], axis=0) # 0.0 std dev
                    segmins = ys - np.std(blockdata[:,0], axis=0) # 0.0 std dev
            ax.fill_between(xs, segmaxs, segmins, color='gray', alpha=0.3, label="Std. Dev. across trajectory blocks")
            ax.set_title("Segment %d-%d" % (segdata[0], segdata[1]), fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xscale('log')
            ax.set_yticks(np.arange(0.0, 1.2, 0.2))

            if overlay_ys is not None:
                ax.plot(xs, overlay_ys, marker='^', color='blue', linewidth=1.5, linestyle=':', label="Experimental")
            ax.legend(fontsize=6)

        def _plot_pdf_pages_df(startslice, endslice):
###         subplot2grid implementation?
#            fig = plt.Figure(figsize=(8.5, 11)) # Letter
#            axis_idxs = []
#            for row in range(4):
#                for col in range(2):
#                    axis_idxs.append((row, col))
#            data_and_axes = zip(data_slice, axis_idxs):
#            d1 = data_and_axes.pop(0)
#            _plot_df_curve(
#            d2 = data_and_axes.pop(0)
#            ax1 = plt.subplot2grid((4,2),d1[1])
#            _plot_df_curve(ax1, d1[0]
#            ax7 = plt.subplot2grid((4,2),d7[1], sharey=ax8)

            fig1, axs1 = plt.subplots(ncols=4, nrows=3, sharex=True,
                                     sharey=True, figsize=(11, 8.5)) # Letter
            fig1.suptitle("Deuterated fractions against time")
            for ax in axs1[:,0]:
                ax.set_ylabel("Deuterated fraction", fontsize=12)
            for ax in axs1[-1,:]:
                ax.set_xlabel("Time / min", fontsize=12)
            axs1 = axs1.flatten()
            if self.avail['_expt_overlay']:
                for a, predsegs, sliceidx, expt in zip(axs1,
                                             np.hstack((self.results.segres['segres'], self.results.c_segfracs[-1]))[startslice:endslice+1],
                                             range(startslice,endslice+1),
                                             self.results.expfracs[startslice:endslice+1]):
                    _plot_df_curve(a, predsegs, self.results.segfracs[:,sliceidx], overlay_ys=expt)
            else:
                for a, predsegs, sliceidx, in zip(axs1,
                                       np.hstack((self.results.segres['segres'], self.results.c_segfracs[-1]))[startslice:endslice+1],
                                       range(startslice,endslice+1)):
                    _plot_df_curve(a, predsegs, self.results.segfracs[:,sliceidx])

            fig2, axs2 = plt.subplots(ncols=4, nrows=3, sharex=True,
                                     sharey=True, figsize=(11, 8.5)) # Letter
            fig2.suptitle("Deuterated fractions against time (log-scaled)")
            for ax in axs2[:,0]:
                ax.set_ylabel("Deuterated fraction", fontsize=12)
            for ax in axs2[-1,:]:
                ax.set_xlabel("Time / min (log-scaled)", fontsize=12)
            axs2 = axs2.flatten()
            if self.avail['_expt_overlay']:
                for a, predsegs, sliceidx, expt in zip(axs2,
                                             np.hstack((self.results.segres['segres'], self.results.c_segfracs[-1]))[startslice:endslice+1],
                                             range(startslice,endslice+1),
                                             self.results.expfracs[startslice:endslice+1]):
                    _plot_log_df_curve(a, predsegs, self.results.segfracs[:,sliceidx], overlay_ys=expt)
            else:
                for a, predsegs, sliceidx in zip(axs2,
                                       np.hstack((self.results.segres['segres'], self.results.c_segfracs[-1]))[startslice:endslice+1],
                                       range(startslice,endslice+1)):
                    _plot_log_df_curve(a, predsegs, self.results.segfracs[:,sliceidx])

            return fig1, fig2

        with PdfPages(self.results.params['outprefix']+"df_curves.pdf") as pdf:
            pages = int(len(self.results.c_segfracs[-1]) / 12) + 1 # Ceiling
            logfigs = []
            try:
                for pg in range(1, pages+1):
                    currfig, logfig = _plot_pdf_pages_df(12*(pg-1), 12*pg)
                    logfigs.append(logfig)
                    pdf.savefig(currfig)
                    plt.close()
            except IndexError:
                currfig, logfig = _plot_pdf_pages_df(12*(pg-1), len(self.results.c_segfracs[-1])) 
                logfigs.append(logfig)
                pdf.savefig(currfig)
                plt.close()
            for currlogfig in logfigs:
                pdf.savefig(currlogfig)
                plt.close()


    def df_convergence(self):
        """Plot the running average of predicted deuteration fractions for each segment.
           
           Plots are saved to a multi-page PDF file df_convergence.pdf, with one
           segment per page.""" 

        def _plot_df_convergence(block_fracs, cumul_fracs, block_errs, seg):
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            fig.suptitle("Convergence of predicted fractions across trajectory")
            ax = fig.gca()
            ax.set_title("Segment %s-%s" % (seg[0], seg[1]))
            xs = self.results.c_n_frames
            for timeidx in range(len(self.results.params['times'])):
                l = ax.plot(xs, cumul_fracs[:, timeidx], label="Time %s min, blocks +/- std. err." % self.results.params['times'][timeidx])
                ax.errorbar(xs, block_fracs[:, timeidx], yerr=block_errs[:, timeidx], fmt='o', capsize=2, color=l[-1].get_color())
            ax.set_ylim(0.0, 1.25) # Space for legend
            ax.set_yticks(np.arange(0.0,1.2,0.2))
            blocksize = self.results.n_frames[0]
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], blocksize)
            ax.set_xticks(xticknums)
            ax.set_xlim(0, xs[-1] * 1.05) 
            ax.set_xlabel("Trajectory frame") 
            ax.set_ylabel("Deuterated fraction") 
            ax.legend(loc='upper center')
            return fig
                
        with PdfPages(self.results.params['outprefix']+"df_convergence.pdf") as pdf:
            for i, currseg in enumerate(self.results.segres['segres']):
                currfig = _plot_df_convergence(self.results.segfracs[:,i,:],
                                               self.results.c_segfracs[:,i,:],
                                               self.results.seg_SEMs[:,i,:],
                                               currseg)
                pdf.savefig(currfig)
                plt.close()

    def seg_curve(self):
        """Plot by-segment deuterated fractions at a given timepoint.
           
           Plots are saved to a multi-page PDF file seg_curves.pdf, with one
           timepoint per page and a final plot of all timepoints. Plots are
           optionally overlaid with experimental curves, according to the 
           value of Plots.avail['_expt_overlay'].""" 

        def _plot_seg_curve(ax, cumul_fracs, seglist, blocksize, time, overlay_fracs=None):
            xs = range(1,len(seglist)+1)
            labels = [ str(i[0])+"-"+str(i[1]) for i in seglist ]
            ax.set_title("Time = %s min, block size = %d" % (time, blocksize))
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation='vertical')
            ax.set_ylabel("Deuterated fraction")
            ax.set_xlabel("Peptide segment")
            ax.set_ylim(0.0, 1.0)

            if overlay_fracs is not None:
                ax.plot(xs, cumul_fracs,
                        label="Predicted fraction, R = %3.2f" % self.results.correls[timeidx])
                ax.plot(xs, overlay_fracs,
                        label="Experimental fraction", linestyle=':')
                fig.suptitle("By-segment predicted & experimental deuterated fractions")
            else:
                ax.plot(xs, cumul_fracs, label="Predicted fraction")
                fig.suptitle("By-segment predicted deuterated fractions")
            return ax

        def _fill_seg_range(ax, seg_fracs, seglist, cumul_fracs=None):
            xs = range(1,len(seglist)+1)
            segmaxs, segmins = np.zeros(seg_fracs.shape[1]), np.zeros(seg_fracs.shape[1])
            # Optional +/- std dev
            for segidx in range(len(segmaxs)):
                with np.errstate(all='raise'):
                    try:
                        segmaxs[segidx] = np.std(seg_fracs[:, segidx], ddof=1) # Pop std. dev.
                        segmins[segidx] = -1 * np.std(seg_fracs[:, segidx], ddof=1) # Pop std. dev.
                    except (FloatingPointError, RuntimeWarning):
                        segmaxs[segidx] = np.std(seg_fracs[:, segidx]) # 0.0 std. dev.
                        segmins[segidx] = -1 * np.std(seg_fracs[:, segidx]) # 0.0 std. dev.
            segmaxs += cumul_fracs
            segmins += cumul_fracs
            ax.fill_between(xs, segmaxs, segmins, color='gray', alpha=0.3, label="Std. Dev. across trajectory blocks")
            return ax

        def _plot_all_seg_curves(cumul_fracs, seglist, times):
            fig = plt.figure(figsize=(11,8.5)) # Letter
            ax = fig.gca()
            xs = range(1,len(seglist)+1)
            labels = [ str(i[0])+"-"+str(i[1]) for i in seglist ]
            ax.set_title("All timepoints (times in min)")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation='vertical')
            ax.set_ylabel("Deuterated fraction")
            ax.set_xlabel("Peptide segment")
            ax.set_ylim(0.0, 1.0)

            for timeidx, t in enumerate(times):
                    ax.plot(xs, cumul_fracs[:, timeidx], label="Predicted @ time %s" % t)
                    fig.suptitle("By-segment predicted deuterated fractions")
            ax.legend()
            fig.tight_layout(rect=[0,0,1,0.95])
            return fig
                    
        with PdfPages(self.results.params['outprefix']+"seg_curves.pdf") as pdf:
            # Single timepoint plots
            for timeidx, t in enumerate(self.results.params['times']):
                fig = plt.figure(figsize=(11, 8.5))
                ax1 = fig.gca()
                if self.avail['_expt_overlay']:
                    fig.suptitle("By-segment predicted & experimental deuterated fractions")
                    ax1 = _plot_seg_curve(ax1, self.results.c_segfracs[-1,:,timeidx], self.results.segres['segres'],
                                          self.results.n_frames[0], t, self.results.expfracs[:,timeidx])
                    # +/- std.dev
                    ax1 = _fill_seg_range(ax1, self.results.segfracs[:,:,timeidx], self.results.segres['segres'],
                                          self.results.c_segfracs[-1,:,timeidx])
                else:
                    fig.suptitle("By-segment predicted deuterated fractions")
                    ax1 = _plot_seg_curve(ax1, self.results.c_segfracs[-1,:,timeidx], self.results.segres['segres'],
                                          self.results.n_frames[0], t)
                    ax1 = _fill_seg_range(ax1, self.results.segfracs[:,:,timeidx], self.results.segres['segres'],
                                          self.results.c_segfracs[-1,:,timeidx])
                ax1.legend()
                fig.tight_layout(rect=[0,0,1,0.95])
                pdf.savefig(fig)
                plt.close()
            # All timepoint plot, no expt.
            currfig = _plot_all_seg_curves(self.results.c_segfracs[-1], self.results.segres['segres'],
                                           self.results.params['times'])
            pdf.savefig(currfig)
            plt.close()

            
    def pf_byres(self):
        """Plot by-residue protection factors averaged over all frames,
           both in raw and log10 scaled forms.
           
           Plots are saved to a multi-page PDF file pf_byres.pdf""" 

        with PdfPages(self.results.params['outprefix']+"pf_byres.pdf") as pdf:
            # Raw PFs
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs = self.results.resnums # ResIDs 
            ax.plot(xs, self.results.c_pfs[-1], linewidth=1)
            ax.set_title("Mean by-residue protection factors")
            ax.set_ylabel("Protection factor")
            ax.set_xlabel("Residue")
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1,2,5,10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 50, mindata=xs[0])
            ax.set_xticks(xticknums) # Residue labels at auto intervals
                                     # skipping last label if it's within 25 of the previous
            ax.set_xlim(0, xs[-1] *1.05)
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()

            # Log PFs
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs = self.results.resnums # ResIDs
            ax.plot(xs, self.results.c_pfs[-1], linewidth=1)
            ax.set_yscale('log')
            ax.set_title("Mean by-residue protection factors (log-scaled)")
            ax.set_ylabel('Protection factor (log-scaled)')
            ax.set_xlabel("Residue")
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1,2,5,10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 50, mindata=xs[0])
            ax.set_xticks(xticknums) # Residue labels at auto intervals
                                     # skipping last label if it's within 25 of the previous
            ax.set_xlim(0, xs[-1] *1.05)
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()

    def tot_pf(self):
        """Plot convergence of total protection factor,
           both in raw and log10 scaled forms.
           
           Plots are saved to a multi-page PDF file tot_pf.pdf""" 

        def _get_ylim(minpf, maxpf):
            maxexp = np.floor(np.log10(maxpf))
            maxy = np.ceil(maxpf/10**maxexp) * 10**maxexp
            minexp = np.floor(np.log10(minpf))
            miny = np.floor(minpf/10**minexp) * 10**minexp
            return miny, maxy

        with PdfPages(self.results.params['outprefix']+"tot_pf.pdf") as pdf:
            # Raw PFs
            startframe, tots = 0, np.zeros(len(self.results.n_frames))
            for i, endframe in enumerate(self.results.c_n_frames):
                tots[i] = stderr(np.sum(self.results.pf_byframe[:,startframe:endframe], axis=0))
                startframe += self.results.n_frames[i]
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs = self.results.c_n_frames
            l = ax.plot(xs, np.sum(self.results.c_pfs, axis=1), label="Running average")
            ax.errorbar(xs, np.sum(self.results.pfs, axis=1), yerr=tots,
                                   label="Block protection factor +/- std. err.", fmt='o', capsize=2, color=l[-1].get_color())
            ax.set_title("Total protection factors across trajectory")
            ax.set_ylabel("Protection factor")
            ax.set_xlabel("Trajectory frame")
            blocksize = self.results.n_frames[0]
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], blocksize)
            ax.set_xticks(xticknums)
            ax.set_xlim(0, self.results.c_n_frames[-1] * 1.05)
            ax.set_ylim(_get_ylim(np.min(np.sum(self.results.pfs, axis=1)), np.max(np.sum(self.results.pfs, axis=1)))) 
            ax.legend()
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()

            # Log PFs
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs = self.results.c_n_frames
            #ax.scatter(xs, np.log10(np.sum(self.results.pfs, axis=1)), label="Block protection factor", marker='o')
            #ax.plot(xs, np.log10(np.sum(self.results.c_pfs, axis=1)), label="Running average")
            l = ax.plot(xs, np.sum(self.results.c_pfs, axis=1), label="Running average")
            ax.errorbar(xs, np.sum(self.results.pfs, axis=1), yerr=tots,
                                   label="Block protection factor +/- std. err.", fmt='o', capsize=2, color=l[-1].get_color())
            ax.set_title("Total protection factors across trajectory (log-scaled)")
            ax.set_ylabel('Protection factor (log-scaled)')
            ax.set_xlabel("Trajectory frame")
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], blocksize)
            ax.set_xticks(xticknums)
            ax.set_xlim(0, self.results.c_n_frames[-1] * 1.05)
            ax.set_yscale('log')
            ax.set_ylim(10**np.floor(np.log10(np.max(np.sum(self.results.pfs, axis=1)))),
                        10**np.ceil(np.log10(np.max(np.sum(self.results.pfs, axis=1))))) 
            ax.legend()
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()

    
    def pf_error(self):
        """Plot convergence of standard errors in total protection factor,
           with respect to block size.

           Measure of suitability of block size choice
           
           Plots are saved to a multi-page PDF file pf_error.pdf""" 

        with PdfPages(self.results.params['outprefix']+"pf_error.pdf") as pdf:
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs, ys = self.results.tot_SEMs[:,0], self.results.tot_SEMs[:,1]
            ax.plot(xs, ys, label="Standard error across block averages", marker='.')
            ax.set_title("Convergence of error in total protection factor with block size")
            ax.set_ylabel('Std. Err.')
            ax.set_xlabel("Block size / frames")
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 100)
            ax.set_xticks(xticknums)
            ax.set_xlim(0, (self.results.c_n_frames[-1] / 2) * 1.05)
            ax.legend()
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()
            
            # Normalised by residue, overlaid with normalised tot_pf
            fig = plt.figure(figsize=(11, 8.5)) # Letter
            ax = fig.gca()
            xs, ys = self.results.norm_res_SEMs[0,:,0], self.results.norm_res_SEMs[0,:,1] 
            ax.plot(xs, ys, label="Individual residue PF normalized std. errors", linewidth=1, alpha=0.2, color='gray')
            for line in self.results.norm_res_SEMs[1:]:
                xs, ys = line[:,0], line[:,1]
                ax.plot(xs, ys, linewidth=1, alpha=0.2, color='gray')
            xs, ys = self.results.norm_tot_SEMs[:,0], self.results.norm_tot_SEMs[:,1]
            ax.plot(xs, ys, label="Total PF normalized std. error", marker='.')
            ax.set_title("Convergence of standard errors normalized to maximum value")
            ax.set_ylabel('Normalized Std. Err.')
            ax.set_xlabel("Block size / frames")
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 100)
            ax.set_xticks(xticknums)
            ax.set_xlim(0, (self.results.c_n_frames[-1] / 2) * 1.05)
            ax.legend()
            fig.tight_layout() # No fig.suptitle = default figure coords
            pdf.savefig(fig)
            plt.close()
        
    def switch_func(self):
        """Plot shape of switching function used for contacts and/or H-bonds
           
           Plots are saved to a multi-page PDF file switching_function.pdf""" 
        smethods = {
                    'rational_6_12' : Functions.rational_6_12,
                    'sigmoid' : Functions.sigmoid,
                    'exponential' : Functions.exponential,
                    'gaussian' : Functions.gaussian
                   }


        # Work out what needs to be added:
        if type(self.results.resobj) == Methods.Radou:
            do_switch = lambda x: smethods[self.results.resobj.params['switch_method']](x, self.results.resobj.params['switch_scale'], self.results.resobj.params['cut_Nc'])
            cutoff_names = ['cut_Nc']
        if type(self.results.resobj) == Methods.PH:
            do_switch = lambda x: smethods[self.results.resobj.params['switch_method']](x, self.results.resobj.params['switch_scale'], self.results.resobj.params['cut_O'])
            cutoff_names = ['cut_O']

        with PdfPages(self.results.params['outprefix']+"switching_function.pdf") as pdf:
            for cutoff_name in cutoff_names:
                fig = plt.figure(figsize=(11, 8.5)) # Letter
                ax = fig.gca()
                xs = np.arange(0, \
                               self.results.resobj.params[cutoff_name] + self.results.resobj.params['switch_width'] + 0.2, \
                               0.005)
                switch_ys = do_switch(xs)
                lowx_idx = int(np.where(xs == self.results.resobj.params[cutoff_name])[0])
                switch_ys[:lowx_idx+1] = 1
                highx_idx = int(np.where(xs == self.results.resobj.params[cutoff_name] + self.results.resobj.params['switch_width'])[0])
                switch_ys[highx_idx+1:] = 0
                
                ax.plot(xs, switch_ys, label="Contact value")
                ax.set_title("Switching function for contacts with cutoff defined by %s" % cutoff_name)
                ax.set_ylabel('Contribution to contact count')
                ax.set_xlabel("Distance / nm")
#            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))
#            xticknums = self._fix_ticks(ax.get_xticks(), xs[-1], 100)
#            ax.set_xticks(xticknums)
#            ax.set_xlim(0, (self.results.c_n_frames[-1] / 2) * 1.05)
                ax.legend()
                fig.tight_layout() # No fig.suptitle = default figure coords
                pdf.savefig(fig)
                plt.close()

    def run(self, **plot_overrides):
        self.choose_plots(**plot_overrides)
        for key, val in self.avail.items():
            if val:
                try:
                    self._funcdict[key]()
                except KeyError:     # For expt/block ave flags
                    continue



### Add further classes below here

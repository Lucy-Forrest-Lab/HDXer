#!/usr/bin/env python

# Transferable functions for HDX analysis

#
import mdtraj as md
import numpy as np
import pickle


# Exception for HDX
class HDX_Error(Exception):
    """Exception in HDX module"""

# Functions
def load_fulltraj(traj, parm, start=1, stop=None, stride=1, standard_names=True, **kwargs):
    """Loads an MDtraj trajectory object with the desired topology
       and coordinates.
       Usage: setup_universe(parm,traj,[start=1,stop=None,stride=1,**kwargs])
       Standard kwargs include atom_indices (an array of 0-indexed
       atoms to keep) and stride (integer of every nth frame to keep).

       'standard_names=False' (not the default here, or in MDTraj) 
       may also be useful for PDB topologies, otherwise amide H might
       be renamed from the atom names provided to the standard PDB identifiers
       (e.g. 'H', 'H2', 'H3' for the terminal NH3 group).
       
       Returns a complete trajectory, which may be memory intensive.

       See also load_trajchunks for an iterative load of large trajectories """
    try:
        parmobj = md.load_topology(parm, standard_names=standard_names)
    except TypeError:
        parmobj = md.load_topology(parm) # MDTraj only has standard_names kwarg for certain filetypes
    t = md.load(traj, top=parmobj, **kwargs)
    if stop is None:
        stop = t.n_frames
    return t[start-1:stop:stride] # Start is zero indexed

def load_trajchunks(traj, parm, start=1, stride=1, standard_names=True, **kwargs):
    """Loads a file into a generator of MDtraj trajectory chunks.
       Useful for large/memory intensive trajectory files
       Usage: load_trajchunks(traj, parm, [start=1, stride=1, **kwargs])
       Standard kwargs include chunk (size of the trajectory chunks
       to load per iteration), and atom_indices (an array of 0-indexed
       atoms to keep).
       
       'standard_names=False' (not the default here, or in MDTraj)
       may also be useful for PDB topologies, otherwise amide H might
       be renamed from the atom names provided to the standard PDB identifiers
       (e.g. 'H', 'H2', 'H3' for the terminal NH3 group). 
   
       Returns a generator object with trajectory iterations."""
    try:
        parmobj = md.load_topology(parm, standard_names=standard_names)
    except TypeError:
        parmobj = md.load_topology(parm) # MDTraj only has standard_names kwarg for certain filetypes
    return md.iterload(traj, top=parmobj, skip=start-1, stride=stride, **kwargs) # Start is zero indexed

def itertraj_slice(gen, chunk, end, stride=1):
    """Slices a generator (returned by load_trajchunks) of size chunk
       to stop after a given number of frames. Ending frame should be
       given with reference to the ORIGINAL trajectory, not the trajectory
       resampled at interval traj[::stride]. This is consistent with the
       'skip' kwarg for mdtraj.iterload
        
       Usage: slice_itertraj(gen, chunk, end, [stride=1])
       Yields: Trajectories of size chunk until original trajectory file
               is exhausted or frame end is reached."""
    yielded_frames = 0                                          
    end /= stride
    end = int(end) # floor
    while yielded_frames + chunk < end:
        yielded_frames += chunk
        x = next(gen)
        yield x
    x = next(gen)
    yield x[:(end - yielded_frames)]
    #raise StopIteration # RuntimeError in 3.7+
    return

def select(traj, selection):
    """Strips a trajectory based on the MDTraj-format text selection
       provided. By default this is "all"
 
       Usage: select(traj, selection)
       Returns: Trajectory with selected atoms"""
    if selection == "all":
        return traj
#    try:
    atms = traj.topology.select(selection)
    new_t = traj.atom_slice(atms)
    # Workaround as atom indices are not renumbered in mdtraj.Topology.subset
    # But are in mdtraj.Topology.copy
    _tmptop = traj.topology.subset(atms)
    new_t.topology = _tmptop.copy()
    return new_t
#    except (ValueError, AttributeError):
#        raise HDX_Error("Your selection of trajectory atoms hasn't been parsed properly - check the syntax")
#    except IndexError:
#        raise HDX_Error("You're attempting to select trajectory atoms that don't exist - check the syntax")

def list_prolines(traj, log="HDX_analysis.log"):
    """Creates a list of proline residues and appropriate resids.
       Resids are output to HDX_analysis.log file by default.

       Usage: list_prolines(traj, [log])
       Returns: Numpy array of [[Proline_ID, Proline_index]]"""
    prolist = [ r.resSeq for r in traj.topology.residues if r.name=='PRO' ]
    proidx = [ r.index for r in traj.topology.residues if r.name=='PRO' ]
    if len(prolist) > 0:
        with open(log, 'a') as f:
            f.write("Prolines identified at resid:\n"+ \
                    "%s\n" % ' '.join(str(i) for i in prolist))
        return np.asarray(list(zip(prolist, proidx)))
    else:
        with open(log, 'a') as f:
            f.write("No prolines found in topology.\n")
        return None


def select_residxs(traj, reslist, protonly=True, invert=False):
    """Returns atom indices of atoms belonging to residues (0-indexed)
       in the supplied list,

       Options to restrict the selection to protein-only atoms
       (default) and/or select all atoms NOT in residues in the supplied list.
       (inversion of selection, off by default)

       Usage: select_resids(traj, reslist, [protonly, invert])
       Returns: Numpy array of selected atom indices"""

    # The topology.select syntax is more restrictive than MDAnalysis here
    # - use list comprehensions instead
    if invert:
        if protonly:
            return np.asarray([ atom.index for atom in traj.topology.atoms if (atom.residue.is_protein and atom.residue.index not in reslist) ])
        else:
            return np.asarray([ atom.index for atom in traj.topology.atoms if (atom.residue.index not in reslist) ])
    elif protonly:
        return np.asarray([ atom.index for atom in traj.topology.atoms if (atom.residue.is_protein and atom.residue.index in reslist) ])
    else:
        return np.asarray([ atom.index for atom in traj.topology.atoms if (atom.residue.index in reslist) ])


def extract_HN(traj, prolines=None, atomselect="(name H or name HN)", log="HDX_analysis.log"):
    """Returns a list of backbone amide H atom indices, suitable
       for use with 'calc_contacts'. Optionally takes an array of 
       resids/indices to skip (normally prolines) and by default returns
       atom indices matching 'name H and backbone'
       
       Usage: extract_NH(traj, [prolines, atomselect])"""

    # Combine res name & ID to concatenated identifier
    atm2res = lambda _: traj.topology.atom(_).residue.name + str(traj.topology.atom(_).residue.resSeq)

    if prolines is not None:
        # Syntax = "... and not (residue 1 or residue 2 or residue 3 ... )"
        atomselect += " and not (residue %s" % ' or residue '.join(str(_) for _ in prolines[:,0]) + ")"
        with open(log, 'a') as f:
            f.write("Extracted HN from resids:\n"+ \
                    "%s\n" % '\n'.join(atm2res(i) for i in traj.topology.select(atomselect)))
        return traj.topology.select(atomselect)
    else:
        with open(log, 'a') as f:
            f.write("Extracted HN from resids:\n"+ \
                    "%s\n" % '\n'.join(atm2res(i) for i in traj.topology.select(atomselect))) 
        return traj.topology.select(atomselect)

### Switching functions for contacts etc. calculation
### 'Sigmoid': y = 1 / [ 1 + exp( -k * (x - d0) ) ]
def sigmoid(x, k=1., d0=0):
    denom = 1 + np.exp( k * (x - d0) )
    return 1./denom # Height 1 = d0 @ midpoint 0.5 contacts, Height 2 = d0 @ midpoint 1.0 contacts

### 'Rational_6_12': y = [ 1 - ( (x - d0) / x0 ) ** n ] / [ 1 - ( (x - d0) / x0 ) ** m ]
def rational_6_12(x, k, d0=0, n=6, m=12):
    num = 1 - ( (x-d0) / k ) ** n
    denom = 1 - ( (x-d0) / k ) ** m
    return num/denom

### 'Exponential': y = exp( -( x - d0 ) / x0 )
def exponential(x, k, d0=0):
    return np.exp( -(x-d0) / k )

### 'Gaussian': y = exp( -( x - d0 )**2 / 2*x0**2 )
def gaussian(x, k, d0=0):
    num = -1 * (x - d0)**2
    denom = 2 * k**2
    return np.exp( num / denom )

### Pickling
def cacheobj(cachefn=None):
    def pickle_decorator(func):
        def pickle_wrapped_func(*args,**kwargs):
            try:
                fn = args[0].params['outprefix'] + kwargs.pop('cachefn')
                cached_obj = pickle.load(open(fn,'rb'))
                try:
                    # args[0] is 'self' for class methods
                    with open(args[0].params['logfile'],'a') as f:
                        f.write("Read cache from file %s\n" % fn)
                except KeyError:
                    print("Read cache from file %s\n" % fn)
                return cached_obj

            except (KeyError, FileNotFoundError, EOFError, TypeError):
                new_obj = func(*args, **kwargs)
            pickle.dump(args[0], open(fn,'wb'), protocol=-1) # Highest protocol for size purposes
            try:
                with open(args[0].params['logfile'],'a') as f:
                    f.write("Saved cache to file %s\n" % fn)
            except KeyError:
                print("Saved cache to file %s\n" % fn)
            return new_obj

        return pickle_wrapped_func
    return pickle_decorator

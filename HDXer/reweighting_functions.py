#!/usr/bin/env python

import numpy as np
import os
from glob import glob
from copy import deepcopy

### 1) Functions for reading in files & run setup ###
# Sorting key for contacts/Hbonds files
def strip_filename(filename, prefix):
    """Sorting key that will strip the integer residue number
       from a Contacts/Hbonds filename. The key splits the filename
       between a provided prefix and the '.' character. Filenames should
       therefore follow the syntax '{prefix}{residue_number}.{suffix}

       Inputs:
           filename (str) : filename to strip
           prefix (str) :  prefix in the filename before the residue number

       Returns:
           residue_number (int)

        Example:
            >>> strip_filename('Contacts_123.tmp', 'Contacts_')
            123
    """
    try:
        _ = filename.split(prefix)[1]
        return int(_.split(".")[0])
    except:
        raise NameError("Unable to read residue number from Contacts/Hbonds file: %s" % filename)


# Read in single column datafiles
def files_to_array(filenames):
    """Read in data fom list of files with np.loadtxt.

       Inputs:
           filenames ( list(str1, str2, ...) ) : files to read in

        Returns:
           Stacked array of the file data ( np.array(n_files, n_lines_per_file) )
    """
    datalist = [ np.loadtxt(file) for file in filenames ]
    try:
        return np.stack(datalist, axis=0)
    except ValueError:
        for filearr, file in zip(datalist, filenames):
            if len(filearr) == 0:
                raise ValueError("Error in stacking files read with np.loadtxt - file %s is empty" % file)
        raise ValueError("Error in stacking files read with np.loadtxt - are they all the same length?")


# Read in initial contacts & H-bonds.
# Store as 2D-np arrays of shape (n_residues, n_frames)

# Treat separate chains as separate frames. Can be upweighted/downweighted individually
# (Only applicable if we add extra lines here to read in the extra Contacts/Hbonds files for each chain)
def read_contacts_hbonds(folderlist, contacts_prefix, hbonds_prefix):
    """Read in contact & hbond files from defined folders & return as numpy arrays,
       along with the resids they correspond to. Filename syntax = {contacts_prefix}*.tmp

       Usage: read_contacts_hbonds(folderlist, contacts_prefix, hbonds_prefix)

       Returns: contacts[n_residues, n_frames], hbonds[n_residues, n_frames], sorted_resids_per_folder"""

    # Turn a single contacts/hbond prefix into a non-string iterable
    if isinstance(contacts_prefix, str):
        contacts_prefix = [ contacts_prefix ]
    if isinstance(hbonds_prefix, str):
        hbonds_prefix = [ hbonds_prefix ]

    # Sort contact & Hbondfiles based on 1) their prefix, then 2) their residue numbers
    contactfiles, hbondfiles = [], []
    for current_prefix in contacts_prefix:
        for folder in folderlist:
            contactfiles.append(sorted(glob(os.path.join(folder, current_prefix + "*.tmp")),
                                       key=lambda x: strip_filename(x, prefix=current_prefix)))
    for current_prefix in hbonds_prefix:
        for folder in folderlist:
            hbondfiles.append(sorted(glob(os.path.join(folder, current_prefix + "*.tmp")),
                                     key=lambda x: strip_filename(x, prefix=current_prefix)))

#    assert np.array(contactfiles).shape == np.array(hbondfiles).shape, \
#           "Found an unequal number of Contact and H-bond files/folders. \n" \
#           "There should be the same number of files & folders if they've been generated correctly by running calc_hdx"

    # Get list of residue IDs in each of the folders
    # This is a list comprehension with the try/except for the extra strings
    resids = []
    for folder_files in contactfiles:
        _ = []
        for current_prefix in contacts_prefix:
            for file in folder_files:
                try:
                    _.append(strip_filename(file, prefix=current_prefix))
                except NameError:
                    pass  # E.g. for 2 chain system
            resids.append(_)

    sorted_resids_per_folder = deepcopy(resids)
    sorted_resids_per_folder.sort(key=lambda _: len(_)) # Sort folders based on number of residues
    filters = list(map(lambda _: np.in1d(_, sorted_resids_per_folder[0]), resids))  # Get indices to filter by shortest
    new_resids = []
    for r, f in list(zip(resids, filters)):
        new_resids.append(np.array(r)[f])
    new_resids = np.stack(new_resids)
    if not np.diff(new_resids, axis=0).sum():  # If sum of differences between filtered resids == 0
        pass
    else:
        raise ValueError(
            "Error in filtering trajectories to common residues - do residue IDs match up in your contacts/H-bond files?")

    _contacts = list(
        map(lambda x, y: x[y], [files_to_array(curr_cfiles) for curr_cfiles in contactfiles], filters))
    _hbonds = list(
        map(lambda x, y: x[y], [files_to_array(curr_hfiles) for curr_hfiles in hbondfiles], filters))

    contacts = np.concatenate(_contacts, axis=1)
    print("Contacts read")
    hbonds = np.concatenate(_hbonds, axis=1)
    print("Hbonds read")
    assert (contacts.shape == hbonds.shape)
    return contacts, hbonds, sorted_resids_per_folder


    # Read intrinsic rates, multiply by times
def read_kints_segments(kintfile, expt_path, n_res, times, sorted_resids):
    """Read in intrinsic rates, segments, and expt deuterated fractions.
       All will be reshaped into 3D arrays of [n_segments, n_residues, n_times]
       Intrinsic rates will be converted to minuskt, i.e. the numerator in the
       rate calculation: exp(-kt / PF)

       Requires number of residues, times, and a list of residue IDs
       for which contacts/H-bonds have been read in

       Usage: read_kints_segments(kintfile, expt_path, n_res, times, sorted_resids)

       Returns: minuskt, expt_dfrac, segfilters (all of shape [n_segments, n_residues, n_times])"""

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

    assert all((segfilters.shape == exp_dfrac.shape,
                segfilters.shape == kint.shape,
                n_res == len(final_resid))) # Check we've at least read in the right number!

    print("Segments and experimental dfracs read")
    return -kint, exp_dfrac, segfilters # Returns minuskt




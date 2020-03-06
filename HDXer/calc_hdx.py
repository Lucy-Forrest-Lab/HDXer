#!/usr/bin/env python

# Wrapper script for HDX analyses
# Author: Richard Bradshaw, richard.bradshaw@nih.gov
#
# For help/usage instructions: calc_hdx.py -h
#
#
# Dependencies
import mdtraj as md
import sys, ast
import argparse
from functools import reduce
# 
import Functions, Methods, Analysis


### Argparser ###
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--traj",help="Trajectory/ies for analysis",nargs='+',type=str, required=True)
    parser.add_argument("-p","--parm",help="Topology file to be used for analysis",type=str, required=True)
    parser.add_argument("-s","--start",help="Frame at which to start (inclusive) reading each trajectory. Default = 1 (every frame)", type=int, default=1)
    parser.add_argument("-e","--end",help="Frame at which to end (inclusive) reading each trajectory. Default = final frame", type=int)
    parser.add_argument("-str","--stride",help="Stride at which to read the trajectory. Default = 1 (every frame)", type=int, default=1)
    parser.add_argument("-c","--chunks",help="If set, trajectory will be read in chunks of this size (lowers memory requirements for large trajectories). Default = 1000", nargs='?', type=int, const=1000)
    parser.add_argument("-sel","--select",help="MDTraj format selection string for atoms to select for analysis from trajectories. Default = 'all'", default='all')
    parser.add_argument("-m","--method",help="Method for analysis. Currently choose between 'Radou' or 'PerssonHalle'", choices=['Radou', 'PerssonHalle'], default='Radou', required=True)
    parser.add_argument("-dt","--times",help="Times for analysis, in minutes. Defaults to [ 0.167, 1.0, 10.0, 120.0 ]", nargs='+', default=[0.167, 1.0, 10.0, 120.0], type=float)
    parser.add_argument("-log","--logfile",help="Name of logfile for printout of run info. Defaults to 'HDX_analysis.log'", type=str, default='HDX_analysis.log')
    parser.add_argument("-seg","--segfile",help="Name of file with segment definitions for analysis. Segments should be defined one per line, with starting/finishing residues whitespace separated. Defaults to 'segfile.txt'",type=str, default='segfile.txt')
    parser.add_argument("-exp","--expfile",help="Name of file with experimental deuterated fractions. Segments should be identical to those in segfile, defined one per line, followed by one column for each timepoint in --times. Whitespace separated. No default.")
    parser.add_argument("-out","--outprefix",help="Prefix for prediction output files",type=str, default='')
    parser.add_argument("-mopt","--method_options",help="Additional method options. Should be provided as a single string in Python dictionary format, e.g.:  '{ 'hbond_method' : 'contacts', 'cut_Nc' : 0.70, 'save_detailed' : True }' (Note the string must be enclosed in quotes)",type=str)
    parser.add_argument("-aopt","--analysis_options",help="Additional analysis options. Should be provided as a single string in Python dictionary format, e.g.:  '{ 'figs' : True }' (Note the string must be enclosed in quotes)",type=str)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.method_options is not None:
        try:
            optdict = ast.literal_eval(args.method_options)
            if isinstance(optdict, dict):
                args.method_options = optdict
            else:
                raise Functions.HDX_Error("Your options flag isn't a dictionary. Dictionary format with key/value pairs is required")
        except ValueError:
            raise Functions.HDX_Error("There's something wrong with the syntax of your options flag. Check it's formatted like a Python dictionary")
    return args

### Main prediction functions ###
def _get_method(name):
    """Choose a method to run based on a string"""
    # Switch for methods (add new ones here):
    methods = { 'radou' : Methods.Radou,
                'perssonhalle' : Methods.PH }

    return methods[name.lower()]
    
def _update_options(opts, **updates):
    """Update options dictionary with extra kwargs"""
    opts.update(updates)

def predict(traj, method, mopts, aopts, saveprefix=None):
    """Predicts fraction of deuterium exchange for residues in the given
       trajectory, using the given method and dictionary of options, and
       creates a suitable Analysis.Analyze object to store these results.
 
       Usage: predict(traj, method, method_options, analysis_options)
       Returns: (Object of desired method class with completed HDX predictions,
                 Analyze object with completed HDX results)"""

    if saveprefix is not None:
        if predict.calls == 1:
            cachefn = saveprefix + '.pkl'
            predict.calls += 1
        else:
            cachefn = saveprefix + '_chunk_%d.pkl' % predict.calls
            predict.calls += 1
    m_obj = _get_method(method)(**mopts)
    m_obj = m_obj.run(traj, cachefn=cachefn)
    a_obj = Analysis.Analyze(m_obj, m_obj.top, **aopts)
    return m_obj, a_obj

def combine_results(first, second):
    """Sum objects in given tuples using the __add__ method of
       each object. e.g.
       
       foo[0] = foo[0] + bar[0]
       foo[1] = foo[1] + bar[1]
       return foo

       Used here to sum Method and Analyze objects together for trajectories
       read in chunks.

       Usage: combine_results( (Method1, Analysis1), (Method2, Analysis2) )
       Returns: (summed_methods, summed_analyses)"""

    comb = [ a + b for a, b in zip(first, second) ]
    return tuple(comb)

def full(trajlist, parm, start, stop, stride, select, method, mopts, aopts, saveprefix):
    """Loads all trajectories in the given list and performs HDX predictions.
       'slicelist' is a list of indices for slicing the trajectory

       Usage: full(trajlist, parm, slicelist, method, method_options, analysis_options)
       Returns: (Object of desired method class with completed HDX predictions,
                 Analyze object with completed HDX results)"""

    predict.calls = 1
    t = Functions.load_fulltraj(trajlist, parm=parm, start=start, stop=stop, stride=stride)
    tslice = Functions.select(t, select)
    summed_results, summed_analysis = predict(tslice, method, mopts, aopts, saveprefix)
    return summed_results, summed_analysis

def chunks(trajlist, parm, start, stop, stride, select, chunksize, method, mopts, aopts, saveprefix):
    """Load trajectories in the given list in chunks and perform HDX predictions.

       Usage: chunks(trajlist, parm, stride, chunksize, method, method_options, analysis_options)
       Returns: (Object of desired method class with completed HDX predictions summed over all frames,
                 Analyze object with completed HDX results by chunk and cumulatively)"""

    predict.calls = 1
    fulllist = []
    for t in trajlist:
        if stop is None:
            final_frame = sum(_.n_frames*stride for _ in Functions.load_trajchunks(t, parm=parm, stride=stride, chunk=chunksize))
        else:
            final_frame = stop
        t_gen = Functions.load_trajchunks(t, parm=parm, start=start, stride=stride, chunk=chunksize)
        f_to_yield = final_frame - (start - 1)
        # Sums generator with __add__ of desired method
        fulllist.append(reduce(combine_results,
                               (predict(Functions.select(t_chunk, select), method, mopts, aopts, saveprefix) for t_chunk in \
                               Functions.itertraj_slice(t_gen, chunksize, f_to_yield, stride=stride))))
    
    resultlist = [ tup[0] for tup in fulllist ]
    analysislist = [ tup[1] for tup in fulllist ]
    firstresult = resultlist.pop(0)
    firstanalysis = analysislist.pop(0)
    return sum(resultlist, firstresult), sum(analysislist, firstanalysis)


### Main below here
if __name__ == '__main__':
    global args    
    args = parse()
    ### Write CL options
    with open(args.logfile, 'a') as f:
        f.write("Command-line arguments: "+' '.join(i for i in sys.argv)+'\n')
    ### Set up options
    if args.method_options is not None:
        _update_options(args.method_options, logfile=args.logfile,
                        segfile=args.segfile, outprefix=args.outprefix,
                        times=args.times)
    else:
        args.method_options = {}
        _update_options(args.method_options, logfile=args.logfile,
                        segfile=args.segfile, outprefix=args.outprefix,
                        times=args.times)
    if args.analysis_options is not None:
        _update_options(args.analysis_options, logfile=args.logfile,
                        segfile=args.segfile, expfile=args.expfile,
                        outprefix=args.outprefix, times=args.times)
    else:
        args.analysis_options = {}
        _update_options(args.analysis_options, logfile=args.logfile,
                        segfile=args.segfile, expfile=args.expfile,
                        outprefix=args.outprefix, times=args.times)
    # Prediction & creation of analysis objects
    if args.chunks is not None:
        results, analysis = chunks(args.traj, args.parm, args.start, args.end,
                         args.stride, args.select, args.chunks,
                         args.method, args.method_options, args.analysis_options, saveprefix='results')
    else:
        results, analysis = full(args.traj, args.parm, args.start, args.end,
                       args.stride, args.select, args.method,
                       args.method_options, args.analysis_options, saveprefix='results')
    # Automatic basic plotting
    if args.chunks is not None:
        # Switch here for methods that don't have a meaningful 'by-frame' PF estimation
        # These require analysis/plotting over the sum total of ALL chunks
        if args.method in ['PerssonHalle']:
            try:
                summed_analysis = Analysis.Analyze(results, results.top, **args.analysis_options) 
            except AttributeError: # 'results' obj may not have topology if it's been deleted for pickling
                results.__setstate__(results.__dict__) # Loads topology.pkl if found
                summed_analysis = Analysis.Analyze(results, results.top, **args.analysis_options) 
            summed_analysis = summed_analysis.run(cachefn='analysis.pkl')
            summed_analysis.print_summaries()
            summed_plots = Analysis.Plots(summed_analysis)
            summed_plots.run()
        else:
            analysis = analysis.run(cachefn='analysis.pkl')
            analysis.print_summaries()
            plots = Analysis.Plots(analysis)
            plots.run()
    else:
        analysis = analysis.run(cachefn='analysis.pkl')
        analysis.print_summaries()
        plots = Analysis.Plots(analysis)
        plots.run()

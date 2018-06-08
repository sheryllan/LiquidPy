import os
import argparse


class TaskBase(object):
    def __init__(self, vollim_dflt, outdir_dflt, outfile_dflt):
        self.vollim_dflt = vollim_dflt
        self.outdir_dflt = outdir_dflt
        self.outfile_dflt = outfile_dflt

    def check(self, vol_threshold, outpath):
        raise NotImplementedError("Please Implement this method")

    def run(self):
        aparser = argparse.ArgumentParser()
        aparser.add_argument('-d', '--outdir', type=str, help='the output directory of the check results')
        aparser.add_argument('-f', '--outfile', type=str, help='the filename of the output results')
        aparser.add_argument('-v', '--vollim', type=int, help='the volume threshold to filter out products')

        args = aparser.parse_args()
        outdir = self.outdir_dflt if args.outdir is None else args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        vollim = self.vollim_dflt if args.vollim is None else args.vollim
        outfile = self.outfile_dflt if args.outfile is None else args.outfile
        outpath = os.path.join(outdir, outfile)
        self.check(vollim, outpath)
        print('results output to {}'.format(outpath))

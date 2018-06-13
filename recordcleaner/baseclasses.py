import os
import argparse


class TaskBase(object):
    OUTPATH = 'outpath'
    VOLLIM = 'vollim'

    def __init__(self, vollim_dflt, outpath_dflt):
        self.vollim_dflt = vollim_dflt
        self.outpath_dflt = outpath_dflt
        self.aparser = argparse.ArgumentParser()
        self.aparser.add_argument('-o', '--' + self.OUTPATH, type=str, help='the output path of the check results')
        self.aparser.add_argument('-v', '--' + self.VOLLIM, type=int, help='the volume threshold to filter out products')

    def check(self, vollim, outpath, **kwargs):
        raise NotImplementedError("Please Implement this method")

    def get_args(self):
        args = vars(self.aparser.parse_args())
        if args[self.OUTPATH] is None:
            args[self.OUTPATH] = self.outpath_dflt
            os.makedirs(os.path.dirname(args[self.OUTPATH]), exist_ok=True)
        if args[self.VOLLIM] is None:
            args[self.VOLLIM] = self.vollim_dflt
        must_args = {self.OUTPATH: args[self.OUTPATH], self.VOLLIM: args[self.VOLLIM]}
        opt_args = {k: args[k] for k in args if k not in must_args and args[k] is not None}
        return must_args, opt_args

    def run(self, **kwargs):
        must_args, opt_args = self.get_args()
        print('Results output to {}'.format(must_args[self.OUTPATH]))
        opt_args.update(kwargs)
        return self.check(must_args[self.VOLLIM], must_args[self.OUTPATH], **opt_args)

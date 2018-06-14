import os
import argparse


class TaskBase(object):
    OUTPATH = 'outpath'
    VOLLIM = 'vollim'

    def __init__(self, settings):
        self.dft_settings = {self.OUTPATH: settings.OUTPATH, self.VOLLIM: settings.VOLLIM}
        self.aparser = argparse.ArgumentParser()
        self.aparser.add_argument('-o', '--' + self.OUTPATH, type=str, help='the output path of the check results')
        self.aparser.add_argument('-v', '--' + self.VOLLIM, type=int, help='the volume threshold to filter out products')
        self.to_icinga = os.getenv(settings.TO_ICINGA, False)
        self.servi


    def check(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

    def get_args(self, **kwargs):
        args = dict(self.dft_settings)
        stdin_args = {k: v for k, v in vars(self.aparser.parse_args()).items() if v is not None and k not in kwargs}
        args.update(stdin_args)
        args.update(kwargs)


        # args = vars(self.aparser.parse_args())
        # if args[self.OUTPATH] is None:
        #     args[self.OUTPATH] = self.outpath_dflt
        #     os.makedirs(os.path.dirname(args[self.OUTPATH]), exist_ok=True)
        # if args[self.VOLLIM] is None:
        #     args[self.VOLLIM] = self.vollim_dflt
        # must_args = {self.OUTPATH: args[self.OUTPATH], self.VOLLIM: args[self.VOLLIM]}
        # opt_args = {k: args[k] for k in args if k not in must_args and args[k] is not None}
        return args

    def run(self, **kwargs):
        args = self.get_args(**kwargs)
        print('Results output to {}'.format(args[self.OUTPATH]))
        results = self.check(**args)
        if self.to_icinga:


    def sendto_icinga(self, results, args):


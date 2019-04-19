import os
import pickle

from os.path import join, exists
class Loader(object):
    def __init__(self, datadir = r'..\data\Oxford_Sorted', outdir = r'output\Oxford_Sorted_or', checkdir='checkpoint'):
        self.datadir = datadir
        self.outdir = outdir
        self.file_lst = []
        self.checkdir = checkdir

    def check_dirpath(self, dirpath):
        if not exists(dirpath):
            os.makedirs(dirpath)

    def collect_filepthlst(self):
        # attention: we collect respective paths, not absolute paths
        # the structure of file list: (filefolder, filename)
        file_lst = []
        for file in os.listdir(self.datadir):
            down_folder = join(self.datadir, file)
            if os.path.isdir(down_folder):
                for down_file in os.listdir(down_folder):
                    down2_folder = join(down_folder, down_file)
                    if os.path.isdir(down2_folder):

                        for down2_file in os.listdir(down2_folder):
                            img_fpath = join(down2_folder, down2_file)
                            if down2_file.startswith('A') and down2_file.endswith('_cal'):
                                file_lst.append((join(file, down_file), down2_file))

        self.file_lst =  file_lst

    def load_checkpoint(self, file_lst_pkl, start_idx_pkl, checkdir=''):
        if checkdir=='':
            this_checkdir = self.checkdir
        else:
            this_checkdir = checkdir
        file_lst = pickle.load(open(join(this_checkdir,file_lst_pkl), 'rb'))
        start_idx = pickle.load(open(join(this_checkdir, start_idx_pkl), 'rb'))
        return file_lst, start_idx











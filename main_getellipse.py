# This file is for collecting the ellipse parameters


from os.path import join
from utils.loader import Loader
import pydicom
import pickle
from utils.correction import *

# Paths
datadir = r'..\data\Oxford_Sorted'
outdir = r'output\Oxford_Sorted_or'
checkdir = r'checkpoint'
file_lst_pkl = 'Oxford_Sorted_file_lst.pkl'
start_idx_pkl = 'Oxford_Sorted_start_indx.pkl'
loader = Loader(datadir, outdir, checkdir)

# Control the procedure
# if we want to change the start index for the procedure
Change_checkpoint_mannually = False
# if we we want to collect the file list and set the start index to 0
Initialization_procedure = False
# if we want to continue the procedure from the start index
Continue_procedure = True

# Main code
if Change_checkpoint_mannually:
    start_idx = 295
    pickle.dump(start_idx, open(join(checkdir, start_idx_pkl), 'wb'))
    print('Change start index to {}.'.format(start_idx))

if Initialization_procedure:
    loader.collect_filepthlst()
    start_idx = 0
    pickle.dump(loader.file_lst, open(join(checkdir,file_lst_pkl), 'wb'))
    pickle.dump(start_idx, open(join(checkdir, start_idx_pkl), 'wb'))
    print('Initialization procedure done.')

if Continue_procedure:
    file_lst, start_idx = loader.load_checkpoint(file_lst_pkl, start_idx_pkl)

    len_filelst = len(file_lst)
    print('We have {} images in total'.format(len_filelst))
    stop_sign = False

    for indx, (file_folder, file_name) in enumerate(file_lst):
        if stop_sign:
            print('We stop, and next time we will start from index: ', str(indx))
            pickle.dump(indx, open(join(checkdir, start_idx_pkl), 'wb'))
            break
        else:
            if indx >= start_idx:
                if indx == len_filelst - 1:
                    print('------------------------------------------')
                    print('Now we deal with the last image, {}th image'.format(indx))
                else:
                    print('------------------------------------------')
                    print('Now we deal with the image: index = {}'.format(indx))
                imgpth = join(datadir, file_folder, file_name)
                ds = pydicom.dcmread(imgpth)
                print('ds.pixel_array.shape ', ds.pixel_array.shape)
                savedir = join(outdir, file_folder)
                loader.check_dirpath(savedir)
                savepath = join(savedir, file_name + '_elpar.json')
                stop_sign = call_interactive_ellipse_correction(ds.pixel_array, savepath)

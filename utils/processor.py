from os.path import join, exists, isfile
import pydicom
import json
import numpy as np
import pandas as pd
import os

class Processor(object):
    def __init__(self, datadir = r'..\data\Oxford_Sorted', outdir = r'output\Oxford_Sorted_or', oldalgodir = r'output\Oxford_Sorted_old_algo'):
        self.datadir = datadir
        self.outdir = outdir
        self.oldalgodir = oldalgodir

    def readpar_oldalgo_onefile(self, file_pth):
        file = open(file_pth, 'r')
        lst = [line.strip().strip('frame 0: ') for line in file]
        reso = eval(lst[0])
        pars = lst[1].split()
        AC, cx, cy, a, b, angle = [eval(par) for par in pars]
        file.close()
        return reso, AC, cx, cy, 2 * a, 2 * b, angle

    def par2array_oldalgo(self, file_lst, standardize=True, save_df=True,
                          col_names=['folder_path', 'file_name','resolution', 'AC', 'center_x', 'center_y', 'width', 'height', 'angle'],
                          save_fname='result_oldalgo.csv', return_colidx=2):
        """
        transform the parameter files to standard array and add the data of resolution and AC
        :param file_lst: (list) The form: (file_folder, file_name)
        :param standardize:
        :param save_df:
        :param col_names:
        :param save_fname:
        :param return_colidx: (int) the start index of column names that we choose to convert to array
        :return: (np.array)
        """
        # stand_arr = np.zeros((len(file_lst), 9))
        stand_pars = pd.DataFrame(columns=col_names)
        for indx, (file_folder, file_name) in enumerate(file_lst):
            GApth = self._searchfile_bycond(join(self.datadir, file_folder), word='GA_')
            ges_age = float(np.loadtxt(GApth))
            non_cal = self.find_nocal(self.datadir, file_folder, file_name)
            par_txt = non_cal + '_res.txt'
            par_txt_pth = join(self.oldalgodir, par_txt)
            reso, AC, cx, cy, width, height, angle = self.readpar_oldalgo_onefile(par_txt_pth)
            angle = angle / np.pi * 180
            if standardize:
                if width < height:
                    width, height = height, width
                    if angle < 0:
                        angle += 90
                    elif angle > 0:
                        angle -= 90

            stand_pars.loc[indx,:] = file_folder, non_cal, reso, ges_age, AC, cx, cy, width, height, angle
        if save_df:
            stand_pars.to_csv(join(self.oldalgodir, save_fname))
            # np.savetxt(join(self.oldalgodir, 'result_oldalgo.txt'), stand_arr)
        stand_arr = stand_pars[col_names[return_colidx:]].values
        return stand_arr

    def _searchfile_bycond(self, path, word='GA_'):
        for filename in os.listdir(path):
            fp = join(path, filename)
            if isfile(fp) and word in filename:
                return fp


    def par2array_grtruth(self, file_lst, standardize=True,
                          save_df=True, col_names=['folder_path', 'file_name', 'resolution', 'ges_age', 'AC', 'center_x', 'center_y', 'width', 'height', 'angle'],
                          save_fname='ground_truth.csv', return_colidx=2):
        """
        transform the parameter files to standard array and add the data of resolution and AC
        :param file_lst: (list) The form: (file_folder, file_name)
        :param standardize:
        :param save_df:
        :param col_names:
        :param save_fname:
        :param return_colidx: (int) the start index of column names that we choose to convert to array
        :return: (np.array)
        """
        # stand_arr = np.zeros((len(file_lst), 9))
        stand_pars = pd.DataFrame(columns=col_names)
        for indx, (file_folder, file_name) in enumerate(file_lst):
            imgpth = join(self.datadir, file_folder, file_name)
            GApth = self._searchfile_bycond(join(self.datadir, file_folder), word='GA_')
            ges_age = float(np.loadtxt(GApth))
            ds = pydicom.dcmread(imgpth)
            delta_x = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
            delta_y = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY

            par_pth = join(self.outdir, file_folder, file_name + '_elpar.json')
            with open(par_pth, 'r') as file:
                cx, cy, width, height, angle = json.load(file)
            if standardize:
                if width < height:
                    width, height = height, width
                    if angle < 0:
                        angle += 90
                    elif angle > 0:
                        angle -= 90
            if delta_x == delta_y:
                AC = 2 * np.pi * np.sqrt((width ** 2 + height ** 2) / 8) * delta_x
                # stand_arr[indx] = file_folder, file_name, delta_x, AC, cx, cy, width, height, angle
                stand_pars.loc[indx,:] = file_folder, file_name, delta_x, ges_age, AC, cx, cy, width, height, angle
            else:
                print('Error! Delta X is not equal to Delta Y, index: ', str(indx))
        if save_df:
            stand_pars.to_csv(join(self.outdir, save_fname))
            # np.savetxt(join(self.outdir, 'ground_truth.txt'), stand_arr)
        stand_arr = stand_pars[col_names[return_colidx:]].values
        return stand_arr

    def check_pars(self, pars, check_range):
        # if the parameters do not meet the standards, then return False
        cx, cy, width, height, angle = pars
        center_check = check_range[0] #((450, 600), (300, 450))
        ab_check = check_range[1] #(300, 500)
        angle_range = check_range[2] #(-90, 90)
        return  not (cx < center_check[0][0] or cx > center_check[0][1] \
            or cy < center_check[1][0] or cy > center_check[1][1] \
            or width < ab_check[0] or width > ab_check[1] \
            or height < ab_check[0] or height > ab_check[1]\
            or angle < angle_range[0] or angle > angle_range[1])

    def find_nocal(self, datadir, file_folder, file_name):
        "find the correspnding no_cal file name of one cal file"
        def get_non_cal(fname, incre_int=10):
            file_code = str(eval(fname[4:11].lstrip('0')) + incre_int)
            n_zero = 7 - len(file_code)
            non_cal = '{}{}{}_nocal'.format(fname[:4], ''.join(['0' for _ in range(n_zero)]), file_code)
            return non_cal

        if exists(join(datadir, file_folder, get_non_cal(file_name, 10))):
            non_cal= get_non_cal(file_name, 10)
        elif exists(join(datadir, file_folder, get_non_cal(file_name, 9))):
            non_cal= get_non_cal(file_name, 9)
        elif exists(join(datadir, file_folder, get_non_cal(file_name, 11))):
            non_cal= get_non_cal(file_name, 11)
        else:
            print('Error! Cannot find corresponding nocal files: for ', file_name)
            non_cal= -1
        return non_cal

    def test_file_order(self, manual_df, my_df):
        # manual_df: (patientID, Exam, ...)
        # my_df: (folder_path, ...)
        def test(x):
            x['folder_path'] = r'{}\{}'.format(x['patientID'], x ['Exam'])
            return x

        manual_new = manual_df.copy()
        manual_new = manual_new.apply(test, axis=1)
        s1 = my_df['folder_path']
        s2 = manual_new['folder_path']
        return s1.eq(s2)




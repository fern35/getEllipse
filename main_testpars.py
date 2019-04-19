# This file is for transform the parameter files to standard array
# and generate evaluations


import seaborn as sns
from os.path import join, exists
from utils.loader import Loader
from utils.processor import Processor
from utils.evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


datadir = r'..\data\Oxford_Sorted'
outdir = r'output\Oxford_Sorted_or'
oldalgodir = r'output\Oxford_Sorted_old_algo'

checkdir = r'checkpoint'
file_lst_pkl = 'Oxford_Sorted_file_lst.pkl'
start_idx_pkl = 'Oxford_Sorted_start_indx.pkl'

col_names = ['folder_path', 'file_name', 'resolution',  'ges_age', 'AC', 'center_x', 'center_y', 'width', 'height', 'angle']
gt_fname = 'ground_truth.csv'
old_fname = 'result_oldalgo.csv'
manual_gt_fname = 'measures_manual_auto.xlsx'

loader = Loader(datadir=datadir, outdir=outdir, checkdir=checkdir)
pr = Processor(datadir=datadir, outdir=outdir, oldalgodir=oldalgodir)
evaluator = Evaluator()

par2stand = True

manual_vs_reclick = False

old_vs_manual= False

old_vs_reclick = False

if par2stand :
    file_lst, start_idx = loader.load_checkpoint(file_lst_pkl, start_idx_pkl)
    ###### load parameters and transform the parameters to standard form:
    grtruth_pars = pr.par2array_grtruth(file_lst=file_lst, save_fname=gt_fname, col_names=col_names)
    ###### load the parameters and transform the parameters to standard form:
    oldalgo_pars = pr.par2array_oldalgo(file_lst=file_lst, save_fname=old_fname, col_names=col_names)


if manual_vs_reclick:
    # Test my ground truth v.s. Manual ground truth
    grtruth_df = pd.read_csv(join(outdir, gt_fname))
    manual_grtruth_df = pd.read_excel(join(datadir, manual_gt_fname), sheet_name='Manual')
    grtruth_df = grtruth_df.groupby('folder_path').mean().reset_index()
    manual_grtruth_df.rename(columns={'Abdominal Circumference': 'AC'}, inplace=True)
    # test if the same order
    # same_order = pr.test_file_order(manual_grtruth_df, grtruth_df)
    evaluator.me_rmse_loa_pars(pars_df=manual_grtruth_df, gt_df=grtruth_df,
                               columns=['AC'], save_fname='ME_RMS_LOA_ManualVsReclick.csv')

if old_vs_manual:

    ####################### pre comparison ###################
    xlsx_file = r'C:\Users\320060127\Documents\Python_Scripts\data\Oxford_Sorted\measures_manual_auto.xlsx'
    grtruth_df_sheet = pd.read_excel(xlsx_file, sheet_name='Manual')
    oldalgo_df_sheet = pd.read_excel(xlsx_file, sheet_name='Auto')
    evaluator.me_rmse_loa_pars(pars_df=oldalgo_df_sheet, gt_df=grtruth_df_sheet,
                               columns=['Abdominal Circumference'],
                               save_fname='ME_RMS_LOA_OldVsManual_pre.csv')
    ####################### my comparison ###################
    oldalgo_df = pd.read_csv(join(oldalgodir, old_fname)).groupby('folder_path').mean().reset_index()
    grtruth_df = grtruth_df_sheet.rename(columns={'Abdominal Circumference': 'AC'})
    evaluator.me_rmse_loa_pars(pars_df=oldalgo_df, gt_df=grtruth_df,
                               columns=['AC'],
                               save_fname='ME_RMS_LOA_OldVsManual_mine.csv')
    evaluator.bland_altman_plot(pars_df=oldalgo_df, gt_df=grtruth_df,
                                columns=['AC'], sep_plots=True,
                                multi_figsize=(12,6))

if old_vs_reclick:
    oldalgo_df = pd.read_csv(join(oldalgodir, old_fname))
    grtruth_df = pd.read_csv(join(outdir, gt_fname))

    # evaluator.dist_pars(pars_df=oldalgo_df, gt_df=grtruth_df,
    #                     columns=col_names[3:], sep_plots=False,
    #                     multi_figsize=(15,6))
    evaluator.dist_diff_pars(pars_df=oldalgo_df, gt_df=grtruth_df,
                        columns=col_names[4:], sep_plots=False,
                        multi_figsize=(15,6))
    mean_dice = evaluator.dice_pars(pars_df=oldalgo_df, gt_df=grtruth_df,
                        columns=col_names[5:])
    print('Mean Dice: ', mean_dice)
    evaluator.bland_altman_plot(pars_df=oldalgo_df, gt_df=grtruth_df,
                                columns=col_names[4:9], sep_plots=False,
                                multi_figsize=(12,6))
    evaluator.me_rmse_loa_pars(pars_df=oldalgo_df, gt_df=grtruth_df,
                               columns=col_names[4:9], save_fname='ME_RMS_LOA_OldVsReclick.csv')

    ###################### AC: average3######################
    oldalgo_df_average3 = oldalgo_df.groupby('folder_path').mean().reset_index()
    grtruth_df_average3 = grtruth_df.groupby('folder_path').mean().reset_index()
    evaluator.me_rmse_loa_pars(pars_df=oldalgo_df_average3, gt_df=grtruth_df_average3,
                               columns=['AC'],
                               save_fname='ME_RMS_LOA_OldVsReclick_average3.csv')


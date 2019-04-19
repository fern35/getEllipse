
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
from shapely.geometry.point import Point
from shapely import affinity
import math

class Evaluator(object):

    def __init__(self, savedir=r'evaluation\Oxford_Sorted_old_algo'):
        self. savedir = savedir

    def dist_pars(self, pars_df, gt_df, columns,
                  save_plots=True, sep_plots=True,
                  multi_figsize=(12,8), multi_figname='Distribution_parameters.jpg'):
        if sep_plots:
            for idx, col in enumerate(columns):
                plt.clf()
                sns.distplot(pars_df[col])
                sns.distplot(gt_df[col])
                plt.title(col, fontsize=10)
                plt.xlabel('values', fontsize=10)
                plt.ylabel('density', fontsize=10)
                plt.legend(['algorithm', 'ground truth'], loc='upper left', fontsize=10)
                if save_plots:
                    plt.savefig(join(self.savedir, 'Distribution_' + col + '.jpg'))
                plt.show()
        else:
            n_row = math.ceil(len(columns) / 3)
            plt.figure(figsize=multi_figsize)
            for idx, col in enumerate(columns):
                plt.subplot(n_row, 3, idx+1)
                sns.distplot(pars_df[col])
                sns.distplot(gt_df[col])
                plt.title(col, fontsize=10)
                plt.xlabel('values', fontsize=10)
                plt.ylabel('density', fontsize=10)
            plt.legend(['algorithm', 'ground truth'], loc='right', fontsize=10)
            plt.tight_layout()
            if save_plots:
                plt.savefig(join(self.savedir, multi_figname))
            plt.show()

    def dist_diff_pars(self, pars_df, gt_df, columns,
                       save_plots=True, sep_plots=True,
                       multi_figsize=(12, 8), multi_figname='Distribution_diff_parameters.jpg'):
        if sep_plots:
            for idx, col in enumerate(columns):
                plt.clf()
                sns.distplot(gt_df[col].values - pars_df[col].values)
                plt.title(col + ': ground truth - prediction', fontsize=10)
                plt.xlabel('values', fontsize=10)
                plt.ylabel('density', fontsize=10)
                if save_plots:
                    plt.savefig(join(self.savedir, 'Distribution_diff_' + col + '.jpg'))
                plt.show()
        else:
            n_row = math.ceil(len(columns) / 3)
            plt.figure(figsize=multi_figsize)
            for idx, col in enumerate(columns):
                plt.subplot(n_row, 3, idx + 1)
                sns.distplot(gt_df[col].values - pars_df[col].values)
                plt.title(col + ': ground truth - prediction', fontsize=10)
                plt.xlabel('values', fontsize=10)
                plt.ylabel('density', fontsize=10)
            plt.tight_layout()
            if save_plots:
                plt.savefig(join(self.savedir, multi_figname))
            plt.show()

    def me_rmse_loa_pars(self, pars_df, gt_df, columns, save_csv=True, save_fname='ME_RMS_LOA.csv'):
        df = pd.DataFrame(columns=columns)

        for idx, col in enumerate(columns):
            e = gt_df[col].values - pars_df[col].values
            me = np.mean(e)
            # mean_gr = np.mean(gt_df[col])
            # me_percent = me / mean_gr
            me_percent = np.mean(e / gt_df[col].values)
            std = np.std(e)
            std_percent = np.std(e / gt_df[col].values)
            # low_limit, high_limit = (me - 1.96 * std) * 100 / mean_gr, (me + 1.96 * std) * 100 / mean_gr
            low_limit, high_limit = (me_percent - 1.96 * std_percent), (me_percent + 1.96 * std_percent)
            rmse = np.sqrt(np.mean(e ** 2))

            df.loc['rmse', col] = rmse
            # df.loc['rmse_percent', col] = rmse_percent * 100
            df.loc['me', col] = me
            df.loc['me_percent', col] = me_percent * 100
            df.loc['low_loa', col] = low_limit * 100
            df.loc['high_loa', col] = high_limit * 100
        if save_csv:
            df.to_csv(join(self.savedir, save_fname))
        return df


    def dice_pars_one(self, x):
        return self._get_dice(x[:5], x[5:])

    def dice_pars(self, pars_df, gt_df, columns, save_plot=True,
                  save_csv=True, save_fname='DICE.csv', save_figname='Distribution_dice.jpg'):
        assert len(columns)==5, 'Error! For computing dice, we need 5 parameters!'
        pars_arr = np.concatenate((pars_df[columns].values, gt_df[columns].values), axis=1)
        dice_arr = np.apply_along_axis(self.dice_pars_one, 1, pars_arr)
        mean_dice = np.mean(dice_arr)
        df = pd.DataFrame(dice_arr, columns=['dice'])
        if save_csv:
            df.to_csv(join(self.savedir, save_fname))

        plt.clf()
        sns.distplot(dice_arr)
        plt.title('dice', fontsize=10)
        plt.xlabel('values', fontsize=10)
        plt.ylabel('density', fontsize=10)
        plt.tight_layout()
        if save_plot:
            plt.savefig(join(self.savedir, save_figname))
        return mean_dice

    def _create_ellipse(self, ell_pars):
        x, y, width, height, angle = ell_pars
        circ = Point((x, y)).buffer(0.5)
        ell = affinity.scale(circ, width, height)
        ellr = affinity.rotate(ell, angle)
        return ellr

    def _get_dice(self, ell1_pars, ell2_pars):
        ellipse1 = self._create_ellipse(ell1_pars)
        ellipse2 = self._create_ellipse(ell2_pars)
        intersect = ellipse1.intersection(ellipse2)
        dice = 2 * (intersect.area) / (ellipse1.area + ellipse2.area)
        return dice

    def bland_altman_plot(self,pars_df, gt_df, columns, save_plot=True, sep_plots=False,
                          multi_figsize=(12, 8), multi_figname='BA_parameters.jpg', *args, **kwargs):
        if sep_plots:
            for col in columns:
                data1 = np.asarray(gt_df[col])
                data2 = np.asarray(pars_df[col])
                mean = np.mean([data1, data2], axis=0)
                diff_percent = (data1 - data2) / data1  # Difference between data1 and data2
                md_percent = np.mean(diff_percent)  # Mean of the difference
                std_percent = np.std(diff_percent, axis=0)  # Standard deviation of the difference

                plt.scatter(mean, diff_percent * 100, *args, **kwargs)
                plt.axhline(0, color='gray', linestyle='--')
                plt.axhline((md_percent + 1.96 * std_percent) * 100, color='gray', linestyle='--')
                plt.axhline((md_percent - 1.96 * std_percent) * 100, color='gray', linestyle='--')
                plt.title(col, fontsize=10)
                plt.xlabel('(gr_truth + algo) / 2)', fontsize=10)
                plt.ylabel('(gr_truth - algo) / gr_truth: %', fontsize=10)
                if save_plot:
                    plt.savefig(join(self.savedir, 'BA_' + col + '.jpg'))
                plt.show()
        else:
            n_row = math.ceil(len(columns) / 3)
            plt.figure(figsize=multi_figsize)
            for idx, col in enumerate(columns):
                plt.subplot(n_row, 3, idx + 1)

                data1 = np.asarray(gt_df[col])
                data2 = np.asarray(pars_df[col])
                mean = np.mean([data1, data2], axis=0)
                diff_percent = (data1 - data2) / data1  # Difference between data1 and data2
                md_percent = np.mean(diff_percent)  # Mean of the difference
                std_percent = np.std(diff_percent, axis=0)  # Standard deviation of the difference

                plt.scatter(mean, diff_percent * 100, *args, **kwargs)
                plt.axhline(0, color='gray', linestyle='--')
                plt.axhline((md_percent + 1.96 * std_percent) * 100, color='gray', linestyle='--')
                plt.axhline((md_percent - 1.96 * std_percent) * 100, color='gray', linestyle='--')
                plt.title(col, fontsize=10)
                plt.xlabel('(gr_truth + algo) / 2)', fontsize=10)
                plt.ylabel('(gr_truth - algo) / gr_truth: %', fontsize=10)
            plt.tight_layout()
            if save_plot:
                plt.savefig(join(self.savedir, multi_figname))
            plt.show()

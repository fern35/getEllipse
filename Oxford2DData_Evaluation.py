# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 17:47:07 2017

@author: 310215777
"""
from math import sqrt, pi
import os
import subprocess
import numpy as np
import xlrd
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom
from PIL import Image

def WriteToExcel(res_dict, xls_filename, lowerBd_dict, upperBd_dict, to_remove):
    workbook = xlsxwriter.Workbook(xls_filename)
    worksheetManual = workbook.add_worksheet("Manual")
    worksheetAuto= workbook.add_worksheet("Auto")
    worksheetDiff= workbook.add_worksheet("Diff")
    red_format = workbook.add_format({'font_color': 'red'})
    
    row = 1
    for patientID in sorted(res_dict):
        for Exam in sorted(res_dict[patientID]):
            worksheetAuto.write(row, 0, patientID)
            worksheetManual.write(row, 0, patientID)
            worksheetDiff.write(row, 0, patientID)
            worksheetAuto.write(row, 1, Exam)
            worksheetManual.write(row, 1, Exam)
            worksheetDiff.write(row, 1, Exam)
            col = 2
            measures_list = []
            for measure_type in sorted(res_dict[patientID][Exam]):
                measures_list.append(measure_type)
                auto_measure = res_dict[patientID][Exam][measure_type]["mean_ok_ROI"]
                manual_measure = res_dict[patientID][Exam][measure_type]["measure"]
                if auto_measure == 0:
                    diff_measure = 10000
                else:
                    diff_measure = manual_measure - auto_measure
                worksheetAuto.write(row, col, auto_measure)
                worksheetManual.write(row, col, manual_measure)
                if diff_measure/manual_measure < lowerBd_dict[measure_type] or diff_measure/manual_measure>upperBd_dict[measure_type]:
                    worksheetDiff.write(row, col, diff_measure/manual_measure, red_format)
                else:                    
                    worksheetDiff.write(row, col, diff_measure)
                col+=1
            row += 1
    
    cur_list = ["patientID", "Exam"] + measures_list
    for col, col_title in enumerate(cur_list):
        worksheetAuto.write(0, col, col_title)
        worksheetManual.write(0, col, col_title)
        worksheetDiff.write(0, col, col_title)
    workbook.close()
    
def PlotBlandAltman(savedir, mean_dict, error_dict):
    lowerBd_dict = {}
    upperBd_dict = {}
    for res_type in mean_dict:
        # Plot Bland-Altman
        error = error_dict[res_type]
        mean = mean_dict[res_type]
        overallStdDiff = np.std(error)
        overallMeanDiff = np.mean(error)
        X2 = mean
        nb_im = len(X2)
        plt.figure(2)
        plt.scatter(X2, error, c='b')
        plt.xlim([np.min(X2), np.max(X2)])
        upperBd = (overallMeanDiff +  1.96 * overallStdDiff) 
        lowerBd = (overallMeanDiff -  1.96 * overallStdDiff) 
        upperBd_dict[res_type] = upperBd
        lowerBd_dict[res_type] = lowerBd
        
        Yline1 = np.ones((nb_im,1),float) * upperBd
        plt.plot(X2, Yline1)
        Yline2 = np.ones((nb_im,1),float) * lowerBd
        plt.plot(X2, Yline2)
        Yzero = np.zeros((nb_im,1),float)
        plt.plot(X2, Yzero, '-', c=[0.5, 0.5, 0.5])  
        str_title = res_type + '\n' + "Confidence Interval {:2.2f}% [ {:2.2f}% ; {:2.2f}% ]".format(overallMeanDiff*100, lowerBd*100,upperBd*100)
        plt.title(str_title)
        plt.xlabel("Manual measure (cm)")
        plt.ylabel("Relative Difference (%) : (Manual - Auto)/Manual")
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, res_type+".png"))
        plt.show()
    return lowerBd_dict, upperBd_dict
    
def ComputeBlandAltmanPercentage(res_dict):
    error_dict = {}
    mean_dict = {}
    for PatientID in res_dict:
        for Exam in res_dict[PatientID]:
            for res_type in res_dict[PatientID][Exam]:
                if res_dict[PatientID][Exam][res_type]["mean_ok_ROI"] != 0:
                    error = (res_dict[PatientID][Exam][res_type]["measure"] - res_dict[PatientID][Exam][res_type]["mean_ok_ROI"])/res_dict[PatientID][Exam][res_type]["measure"]
                    mean = (res_dict[PatientID][Exam][res_type]["measure"] + res_dict[PatientID][Exam][res_type]["mean_ok_ROI"])/2.
                else:
                    continue
                try:
                    error_dict[res_type].append(error)
                    mean_dict[res_type].append(mean)
                except:
                    error_dict[res_type] = [error]
                    mean_dict[res_type] = [mean]
    return error_dict, mean_dict
    
def ComputeBlandAltman(res_dict):
    error_dict = {}
    mean_dict = {}
    for PatientID in res_dict:
        for Exam in res_dict[PatientID]:
            for res_type in res_dict[PatientID][Exam]:
                if res_dict[PatientID][Exam][res_type]["mean_ok_ROI"] != 0:
                    error = res_dict[PatientID][Exam][res_type]["measure"] - res_dict[PatientID][Exam][res_type]["mean_ok_ROI"]
                    mean = (res_dict[PatientID][Exam][res_type]["measure"] + res_dict[PatientID][Exam][res_type]["mean_ok_ROI"])/2.
                else:
                    continue
                try:
                    error_dict[res_type].append(error)
                    mean_dict[res_type].append(mean)
                except:
                    error_dict[res_type] = [error]
                    mean_dict[res_type] = [mean]
    return error_dict, mean_dict

def PlotError(error_dict):
    for res_type in error_dict:
        plt.figure()
        plt.plot(error_dict[res_type], marker = 'x', linewidth = 0)
        plt.title(res_type)
    plt.show()

def ComputeMeanError(res_dict):
    error_dict = {}
    for PatientID in res_dict:
        for Exam in res_dict[PatientID]:
            for res_type in res_dict[PatientID][Exam]:
                if res_dict[PatientID][Exam][res_type]["mean_ok_ROI"] != 0:
                    error = res_dict[PatientID][Exam][res_type]["measure"] - res_dict[PatientID][Exam][res_type]["mean_ok_ROI"]
                else:
                    continue
                try:
                    error_dict[res_type].append(error)
                except:
                    error_dict[res_type] = [error]
    return error_dict

def MixMeasureAndResultDict(measure_dict, res_dict):
    for PatientID in res_dict:
        for Exam in res_dict[PatientID]:
            for res_type in res_dict[PatientID][Exam]:
                res_dict[PatientID][Exam][res_type]["measure"] = measure_dict[PatientID][Exam][res_type]
    return res_dict

def ReadMeasuresFromExcel(xls_filename):
    workbook = xlrd.open_workbook(xls_filename)
    worksheet = workbook.sheet_by_index(0)
    
    num_cols = worksheet.ncols
    num_rows = worksheet.nrows
    
    measure_dict = {}
    
    for col_idx in range(num_cols):
        cell_value = worksheet.cell(0,col_idx).value
        if cell_value == "patientID":
            patientIDColIdx = col_idx
            for row_idx in range(1,num_rows):
                PatientID = worksheet.cell(row_idx,col_idx).value
                measure_dict[PatientID] = {}

    for col_idx in range(num_cols):
        cell_value = worksheet.cell(0,col_idx).value
        if cell_value == "Exam":
            ExamColIdx = col_idx
            for row_idx in range(1,num_rows):
                PatientID = worksheet.cell(row_idx,patientIDColIdx).value
                Exam = worksheet.cell(row_idx,ExamColIdx).value
                measure_dict[PatientID][Exam] = {}
    
    for row_idx in range(1,num_rows):    # Iterate through rows
        for col_idx in range(2, num_cols):  # Iterate through columns
            PatientID = worksheet.cell(row_idx,patientIDColIdx).value
            Exam = worksheet.cell(row_idx,ExamColIdx).value
            MeasureName = worksheet.cell(0,col_idx).value
            MeasureValue = worksheet.cell(row_idx,col_idx).value
            try:
                measure_dict[PatientID][Exam][MeasureName] = MeasureValue
            except:
                measure_dict[PatientID][Exam][MeasureName] = {}
    return measure_dict


def ComputeMeanAndStd(exam_dict, list_to_remove):
    to_keep= []
    for acq in exam_dict:
        if acq[:2] in list_to_remove:
            print (acq[:2])
            return 0,0
        if exam_dict[acq]>0:
            to_keep.append(exam_dict[acq])
    if len(to_keep) == 0:
        return 0,0
    to_keep = np.array(to_keep)
    return np.mean(to_keep), np.std(to_keep)


def AddMeanAndStdToDict(res_dict, to_remove):
    for PatientID in res_dict:
        for Exam in res_dict[PatientID]:
            for res_type in res_dict[PatientID][Exam]:
                try:
                    list_to_remove = to_remove[PatientID][Exam]
                except:
                    list_to_remove = []
                mean_exam, std_exam = ComputeMeanAndStd(res_dict[PatientID][Exam][res_type],list_to_remove)
                res_dict[PatientID][Exam][res_type]["mean_ok_ROI"] = mean_exam
                res_dict[PatientID][Exam][res_type]["std_ok_ROI"] = std_exam
    return res_dict


def GetResultType(acq_type):
    if acq_type == "abdomen":
         res_type = ["Abdominal Circumference", "Tranverse Abdominal Diameter"]
    return res_type


def ReadEllipse(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        resol = float(lines[0])
        split1 = lines[1].split(":")
        split2 = split1[1].split(" ")
        center = np.array([split2[2], split2[3]]).astype(float)
        axis = np.array([split2[4], split2[5]]).astype(float)
        #axis /= resol
        angle = float(split2[6])
    return center, axis, angle


def ReadCallipers(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        split1 = lines[2].split(":")
        split2 = split1[1].split(" ")
        cal1 = np.array([split2[1], split2[2]]).astype(float)
        cal2 = np.array([split2[3], split2[4]]).astype(float)
    return cal1, cal2


def GetEllipse(center, axis, angle):
    a = axis[0]
    b = axis[1]
    theta_range = np.linspace(0,2*pi, num = 2*pi*a, endpoint = True)
    x_ellipse = a * np.cos(theta_range)
    y_ellipse = b * np.sin(theta_range)
    x_ellipse_oriented = x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle)
    y_ellipse_oriented = x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle)
    x_ellipse_centered = x_ellipse_oriented + center[0]
    y_ellipse_centered = y_ellipse_oriented + center[1]
    return x_ellipse_centered, y_ellipse_centered


def GetCallipersFromCenterAxisAngle(center, axis_length, angle, axis_type = "short"):
    if axis_type == "long":
        diff_vect = axis_length*np.array([np.cos(angle), np.sin(angle)])
    elif axis_type == "short":
        diff_vect = axis_length*np.array([np.cos(angle + pi/2), np.sin(angle + pi/2)])
    cal2 = center + diff_vect
    cal1 = center - diff_vect
    return cal1, cal2
        

def PlotCallipers(ax, x1, x2, color = 'y'):
    ax.scatter([x1[0], x2[0]], [x1[1], x2[1]], color = color, marker = 'x')


def PlotEllipse(ax, center, axis, angle, color="y",linestyle="--"):
    x_ellipse, y_ellipse = GetEllipse(center, axis, angle)
    ax.plot(x_ellipse, y_ellipse, color = color, linestyle = linestyle)
    return x_ellipse, y_ellipse


def PlotImg(ax, imfilename):
    ds = pydicom.read_file(imfilename)
    img = ds.pixel_array
    h_im = ax.imshow(img, cmap = cm.Greys_r)
    return h_im


def PlotResult(resfilename, imfilename, acq_type = "abdomen"):
    Exam =  os.path.basename(os.path.dirname(imfilename))
    PatientID =  os.path.basename(os.path.dirname(os.path.dirname(imfilename)))
    imName = os.path.basename(resfilename)[:-8]
    savefilename = os.path.join(os.path.dirname(resfilename) ,PatientID + "_"+Exam+"_"+ imName+ "_test.png")
    f, ax = plt.subplots()
    if acq_type == "abdomen":
        center, axis, angle = ReadEllipse(resfilename)
        cal1, cal2 = GetCallipersFromCenterAxisAngle(center, axis[0], angle, axis_type = "long")
        h_im = PlotImg(ax, imfilename)
        PlotEllipse(ax, center, axis, angle)
        PlotCallipers(ax, cal1, cal2)
    plt.axis("equal")
    ax.set_axis_off()
#    plt.axis(h_im.get_extent())
    plt.tight_layout()
    plt.savefig(savefilename, dpi = 150, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    plt.close()
#    plt.show()


def SaveImgAsPng(resdir, imdir, cur_file):
    num = str(int(cur_file[-9:-6]) - 10)
    if len(num) == 2:
        num = "0"+num  
    elif len(num) == 1:
        num = "00"+num
    imfile = os.path.join(imdir,cur_file[:-9] + num + "_cal")
    ds = pydicom.read_file(imfile)
    img = ds.pixel_array
    img_crop = img[97:669,122:953]
    result = Image.fromarray(img_crop)
    result.save(imfile + ".png")

def SaveResAsDict(basedir, resdir, acqs = ["abdomen"]):
    res_dict = {}
    for root, cur_dir, files in os.walk(basedir):
        for cur_file in files:
            PatientID = os.path.basename(os.path.dirname(root))
            Exam = os.path.basename(root)
            
            for acq_type in acqs:
                type_letter = GetLetterPerType(acq_type)
                res_type = GetResultType(acq_type)
                if cur_file.startswith(type_letter) and "nocal" in cur_file and not ".png" in cur_file and not ".txt" in cur_file:
                    resfile = os.path.join(resdir, cur_file + "_res.txt")
                    result = ReadResult(resfile, acq_type)
                    if result[0] == 0:
                        print (root, cur_file)
                    num = cur_file[1]
                    imfilename = [f for f in files if f[1] == num and "nocal" in f and f.startswith][0]
                    imfile = os.path.join(root,imfilename)
                    PlotResult(resfile, imfile, acq_type)
#                    SaveImgAsPng(root, root, cur_file)
                    for i in range(len(res_type)):
                        try:
                            res_dict[PatientID][Exam][res_type[i]][cur_file] = result[i]
                        except:
                            try:
                                res_dict[PatientID][Exam][res_type[i]] = {}
                                res_dict[PatientID][Exam][res_type[i]][cur_file] = result[i]
                            except:
                                try:
                                    res_dict[PatientID][Exam] = {}
                                    res_dict[PatientID][Exam][res_type[i]] = {}
                                    res_dict[PatientID][Exam][res_type[i]][cur_file] = result[i]
                                except:
                                    res_dict[PatientID] = {}
                                    res_dict[PatientID][Exam] = {}
                                    res_dict[PatientID][Exam][res_type[i]] = {}
                                    res_dict[PatientID][Exam][res_type[i]][cur_file] = result[i]
    return res_dict

                
def ReadResult(resfile, acq_type):
    if acq_type == "abdomen":
        return ReadResultAbdomen(resfile)
    return
        

def ReadResultAbdomen(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        res = float(lines[0])
        split1 = lines[1].split(":")
        split2 = split1[1].split(" ")
        AC = float(split2[1])
        TAD = float(split2[5])*2*res
    return [AC, TAD]
    

def TestExe(exedir, basedir, outdir, acq_type):
    test_file_list = GetTestFileListPerType(basedir, outdir, acq_type)
    for test_file in test_file_list:
        cmd_exe_name = "{}/{}".format(exedir, "AbdomenCirc2D_Test.exe")
        cmd_img_name = "{}/{}/{}/{}".format(basedir,test_file[0], test_file[1],test_file[2])
        cmd_GA_name = "{}/{}/{}/{}".format(basedir,test_file[0], test_file[1],test_file[3])
        cmd_out_name = "{}/".format(outdir)
        print (cmd_exe_name, cmd_img_name, cmd_GA_name, cmd_out_name)
        subprocess.check_call([cmd_exe_name,cmd_img_name,cmd_GA_name,cmd_out_name])


def GetLetterPerType(acq_type):
    return acq_type[0].upper()


def GetTestFileListPerType(basedir, outdir, acq_type = "abdomen"):
    test_file_list = []
    type_letter = GetLetterPerType(acq_type)
    for root, cur_dir, files in os.walk(basedir):
        for cur_file in files:
            if cur_file.startswith(type_letter) and "nocal" in cur_file and not ".png" in cur_file and not ".txt" in cur_file:
                PatientID = os.path.basename(os.path.dirname(root))
                Exam = os.path.basename(root)
                GA_file_text = "GA_" + PatientID + "_" + Exam + ".txt"
                cur_test_file = [PatientID, Exam, cur_file, GA_file_text]
                test_file_list.append(cur_test_file)
    return test_file_list
          

if __name__ == "__main__":
    # BASEDIR = r"C:\Users\frq09335\cciofolo\Data\OxfordHD9\Test"
    BASEDIR = r'..\data\Oxford_Sorted'
    # OUTDIR = r"C:\Users\frq09335\cciofolo\Data\OxfordHD9\Test"
    OUTDIR = r'output\Oxford_Sorted_old_algo'
    # EXEDIR = r"C:\Users\frq09335\cciofolo\Projects\FetalUS2D\Repository\Code\SolutionsForDelivery\20170116\FetalBiometry2D_Delivery_201701-Copy\bin\Win32\Release"
    EXEDIR = r'C:\Users\320060127\Documents\Software\AbdomenCirc2D_Test'
    
    TestExe(EXEDIR, BASEDIR, OUTDIR, "abdomen")

    res_dict = SaveResAsDict(BASEDIR, OUTDIR, ["abdomen"])
    res_dict = AddMeanAndStdToDict(res_dict, [])

    xls_filename = os.path.join(BASEDIR,"measures.xlsx")
    xls_save_filename = os.path.join(BASEDIR,"measures_manual_auto.xlsx")

    measure_dict = ReadMeasuresFromExcel(xls_filename)
    res_dict = MixMeasureAndResultDict(measure_dict, res_dict)
    error_dict, mean_dict = ComputeBlandAltmanPercentage(res_dict)
    lowerBd_dict, upperBd_dict = PlotBlandAltman(BASEDIR, mean_dict, error_dict)
    WriteToExcel(res_dict, xls_save_filename, lowerBd_dict, upperBd_dict, [])

import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection

pd.isna=pd.isnull
np.random.seed(123)

class FeatureColumn:
    def __init__(self, category, field, preprocessor, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessor = preprocessor
        self.args = args
        self.data = None
        self.cost = cost

class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns # Depricated
        self.dataset = None # Depricated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                #df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process' + field)
            df.append(df_col)
        df = pd.concat(df, axis=1)
        #df = pd.merge(df, df_sel, how='outer')
        # do preprocessing steps
        df_proc = []#[df['SEQN']]
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessor is not None:
                prepr_col = fe_col.preprocessor(df[field], fe_col.args)
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset
    
    
# Preprocessing functions
def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_real(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical standardization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col

def preproc_cut(df_col, bins):
    # limit values to the bins range
    df_col = df_col[df_col >= bins[0]]
    df_col = df_col[df_col <= bins[-1]]
    return pd.cut(df_col.iloc[:,0], bins, labels=False)

def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col

#### Add your own preprocessing functions ####

## impute median; log then standardize
#def preproc_median_log(df_col, args=None):
#    if args is None:
#        args={'cutoff':np.inf}
#    # other answers as nan
#    df_col[df_col > args['cutoff']] = np.nan
#    # nan replaced by mean
#    df_col[pd.isna(df_col)] = df_col.median()
#    # log transform
#    df_col=np.log10(df_col)
#    # statistical standardization
#    df_col = (df_col-df_col.mean()) / df_col.std()
#    return df_col
    
# impute median; standardize
def preproc_median(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.median()
    # log transform
    # df_col=np.log2(df_col)
    # statistical standardization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_AnnualIncome(df_col, args=None):
    # limit values to the bins outside of range; 
    # customized for annual household income
    df_col[df_col == 12]=np.nan
    df_col[df_col == 13]=np.nan
    for i in range(len(df_col)):
        if df_col.iloc[i]==14:
            df_col.iloc[i]=11
        elif df_col.iloc[i]==15:
            df_col.iloc[i]=12
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.median()
    
    # statistical standardization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_BloodMetal(df_col, args=None):
    df_col[df_col==0]=np.nan
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.median()
    # statistical standardization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_Herpes(df_col, args=None):
    for i in range(len(df_col)):
        if df_col.iloc[i]==3:
            df_col.iloc[i]=2
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_Smoked100times(df_col, args=None):
    for i in range(len(df_col)):
        if df_col.iloc[i]==7 or df_col.iloc[i]==9:
            df_col.iloc[i]=2
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_SkinReact(df_col, args=None):
    for i in range(len(df_col)):
        if df_col.iloc[i]==77 or df_col.iloc[i]==99:
            df_col.iloc[i]=6
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_FrozenPizza(df_col, args=None):
    for i in range(len(df_col)):
        if df_col.iloc[i]==7777 or df_col.iloc[i]==9999:
            df_col.iloc[i]=np.nan
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.median()
    
    # statistical standardization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

## impute mean; log then standardize 
#def preproc_mean_log(df_col, args=None):
#    if args is None:
#        args={'cutoff':np.inf}
#    # other answers as nan
#    df_col[df_col > args['cutoff']] = np.nan
#    # nan replaced by mean
#    df_col[pd.isna(df_col)] = df_col.mean()
#    # log transform
#    df_col_lst=list(df_col)
#    df_col_log=[]
#    for i in range(0,len(df_col_lst)):
#        if df_col_lst[i]==0:
#            df_col_log.append(0)
#        else:
#            df_col_log.append(np.log2(df_col_lst[i]))
#    # statistical standardization
#    df_col_log=np.array(df_col_log)
#    df_col_log = (df_col_log-df_col_log.mean()) / df_col_log.std()
#    return df_col_log


# Dataset loader
class Dataset():
    """ 
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.costs = None
        self.feat_names = []
        
    #### Add your own dataset loader ####
    def load_arthritis(self, opts=None):
        columns = [
            # TARGET: individual was ever told she/he has cancer
            FeatureColumn('Questionnaire', 'MCQ220', 
                                    None, None),
            ## Demographics
            # Gender (*)
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 preproc_onehot, None),
            # Age at time of screening (*)
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 preproc_median, None),
            # Race/Hispanic origin w/ NH Asian (*)
            FeatureColumn('Demographics', 'RIDRETH3',
                                 preproc_onehot, None),
            # Total number of people in the household
            FeatureColumn('Demographics', 'DMDHHSIZ',
                                 preproc_median, None),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHIN2',
                                 preproc_AnnualIncome, {'cutoff':12}),
            # Ratio of family income to poverty
            FeatureColumn('Demographics', 'INDFMPIR',
                                 preproc_median, {'cutoff':5}),

            ## Examination
            # 60 sec HR
            FeatureColumn('Examination', 'BPXCHR',
                                 preproc_median, None),
            # 60 sec pulse
            FeatureColumn('Examination', 'BPXPLS',
                                 preproc_median, None),
            # Pulse regular or irregular
            FeatureColumn('Examination', 'BPXPULS',
                                 preproc_onehot, None),
            # Max inflation levels
            FeatureColumn('Examination', 'BPXML1',
                                 preproc_median, None),                                         
            # Systolic BP (1st rdg)
            FeatureColumn('Examination', 'BPXSY1',
                                 preproc_median, None),
            # Diastolic BP (1st rdg)
            FeatureColumn('Examination', 'BPXDI1',
                                 preproc_median, None),
            # Weight (*)
            FeatureColumn('Examination', 'BMXWT',
                                 preproc_median, None),
            # BMI (*)(*)
            FeatureColumn('Examination', 'BMXBMI',
                                 preproc_median, None),
            
            ## Laboratory
            # Blood lead (ug/dL)
            FeatureColumn('Laboratory', 'LBXBPB',
                                 preproc_median, None),
            # Blood cadmium (umol/L) (*)
            FeatureColumn('Laboratory', 'LBDBCDSI',
                                 preproc_median, None),
            # Blood mercury (umol/L)
            FeatureColumn('Laboratory', 'LBDTHGSI',
                                 preproc_median, None),
            # Blood selenium (ug/L)
            FeatureColumn('Laboratory', 'LBXBSE',
                                 preproc_median, None),
            # Blood chromium (nmol/L) (*)
            FeatureColumn('Laboratory', 'LBDBCRSI',
                                 preproc_median, None),
            # Blood manganese (umol/L)
            FeatureColumn('Laboratory', 'LBDBMNSI',
                                 preproc_median, None),
            # Urine test for chlamydia 
            FeatureColumn('Laboratory', 'URXUCL',
                                 preproc_onehot, None),
            # High density cholesterol (mg/dL)
            FeatureColumn('Laboratory', 'LBDHDD',
                                 preproc_median, None),
            # Total cholesterol (mg/dL)
            FeatureColumn('Laboratory', 'LBXTC',
                                 preproc_median, None),
            # Oral HPV result
            FeatureColumn('Laboratory', 'ORXHPV',
                                 preproc_Herpes, None),
            # Hep B core antibody (*)(*)
            FeatureColumn('Laboratory', 'LBXHBC',
                                  preproc_onehot, None),
            # Hep B surface antibody
            FeatureColumn('Laboratory', 'LBXHBS',
                                  preproc_onehot, None),
            # Herpes Type 2
            FeatureColumn('Laboratory', 'LBXHE2',
                                  preproc_Herpes, None),
            # Herpes Type 1
            FeatureColumn('Laboratory', 'LBXHE1',
                                  preproc_Herpes, None),
            # HIV 1/2 combo test
            FeatureColumn('Laboratory', 'LBXHIVC',
                                  preproc_onehot, None),
            #Iodine in urine (ng/mL)
            FeatureColumn('Laboratory','URXUIO',
                                  preproc_median, None),
            # Acrylamide (pmol/G Hb)
            FeatureColumn('Laboratory', 'LBXACR',
                                  preproc_median, None),
            # Urine flow rate #1 (mL/min)
            FeatureColumn('Laboratory', 'URDFLOW1',
                                  preproc_median, None),
            
            ## Questionnaire
            # General health condition
            FeatureColumn('Questionnaire', 'HSD010',
                                 preproc_onehot, None),
            # Smoked at least 100 cigarettes in life (*)
            FeatureColumn('Questionnaire', 'SMQ020',
                                 preproc_Smoked100times, None),
            # Avg # of cigarettes/day in past 30 days
            FeatureColumn('Questionnaire', 'SMD650',
                                 preproc_median, {'cutoff':95}),
            # Avg # alcoholic drinks/day - past 12 mos (*)
            FeatureColumn('Questionnaire', 'ALQ130',
                                 preproc_median, {'cutoff':15}),
            # Covered by health insurance
            FeatureColumn('Questionnaire', 'HIQ011',
                                 preproc_Smoked100times, None),
            # Vigorous activities in a typical week
            FeatureColumn('Questionnaire', 'PAQ650',
                                 preproc_Smoked100times, {'cutoff':2}),
            # Skin reaction to sun after non-exposure
            FeatureColumn('Questionnaire', 'DED031',
                                 preproc_SkinReact, None),
            # Number of frozen meals/pizza in past 30 days
            FeatureColumn('Questionnaire', 'DBD910',
                                 preproc_FrozenPizza, None), 
            # Num of times in  past yr you had sunburn (*)
            FeatureColumn('Questionnaire','DEQ038Q',
                          preproc_median, {'cutoff':20}),
            # Minutes spent outdoors on workday (*)
            FeatureColumn('Questionnaire', 'DED120',
                          preproc_median, {'cutoff':480}),
            # Age when first menstrual period occurred (*)
            FeatureColumn('Questionnaire', 'RHQ010',
                          preproc_median, {'cutoff':22}),
            # Doctor ever say you were overweight (*)(*)
            FeatureColumn('Questionnaire', 'MCQ080',
                          preproc_Smoked100times, None), 
            # Ever told you had high blood pressure
            FeatureColumn('Questionnaire', 'BPQ020',
                          preproc_Smoked100times, None), 
            # Ever told you had high cholesterol
             FeatureColumn('Questionnaire', 'BPQ080',
                          preproc_Smoked100times, None),
            # Had both ovaries removed
            FeatureColumn('Questionnaire', 'RHQ305',
                          preproc_Smoked100times, None),
            # Ever took estrogen/progresterone combo pills
            FeatureColumn('Questionnaire', 'RHQ570',
                          preproc_Smoked100times, None),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ220'], axis=1)
        features = fe_cols.values
        target = df['MCQ220'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        targets = np.zeros(target.shape[0])
        targets[target == 1] = 1 # yes arthritis # switched these
        targets[target == 2] = 0 # no arthritis

       # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
        self.feat_names = fe_cols.columns

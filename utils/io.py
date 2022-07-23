import os
import ast
import csv
import glob
import pickle
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from biosppy.signals import tools
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def load_dataset(path, sampling_rate, release=False):

    if path.split('/')[-2] == 'ptbxl':
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'ICBEB':
        # load and convert annotation data
        Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb


def load_raw_data_icbeb(df, sampling_rate, path):
    
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
            
    return data

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def load_formmated_raw_data(inputfolder, task, outputfolder, sampling_frequency=500):

    # Load PTB-XL data
    data,raw_labels = load_dataset(inputfolder, sampling_frequency)
    
    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, inputfolder, task)
        
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(
        data, labels, task, min_samples=0, outputfolder=outputfolder)
    
    return data, Y, labels, _.classes_


def load_snippet_data(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    return X, Y

def load_snippet_data_with_il(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']

    return X, Y, I, L


def load_snippet_data_with_il_info(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']
    
    info = data['info']

    return X, Y, I, L, info


def load_state_data(inputfolder):

    data_dict = load_pkfile(inputfolder)

    return data_dict

def load_csv(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar = '|')
        #next(spamreader)
        for row in spamreader:
            data.append(row[0].split(":"))
    return data


def load_label(filepath):
    refernces = dict()
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar = '|')
        next(spamreader)
        for row in spamreader:
            refernces[row[0]] = row[1:]
    return refernces


def load_raw_data(labelpath, filepath, headerpath, filelist):
    raw_data = []
    raw_labels = []
    raw_labellist = []
    raw_headers = []
    refernces = load_label(labelpath)
    file_list = glob.glob(headerpath)
    
    text_file = open(filelist, "r")
    file_list = text_file.read().split(',')
    
    for file in tqdm(file_list):
        header = load_csv(os.path.join(filepath,file+".hea"))
        fid = file.split('/')[-1].split('.')[0]
        label = int(refernces[fid][0])-1
        data = scipy.io.loadmat(os.path.join(filepath,file.split('/')[-1].split(".")[0]+".mat"))
        
        raw_data.append(data)
        raw_headers.append(header)
        raw_labels.append(label)
        tmp_list = []
        for l in refernces[fid]:
            if(l is not ""):
                tmp_list.append(int(l)-1)
        raw_labellist.append(tmp_list)


    return raw_data,raw_labels,raw_labellist,raw_headers


def check_folder(path):

    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Create : ", path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        print(path, " exists")

def load_tsv(path):

    data = []

    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    for row in read_tsv:

        data.append(np.array(row).astype(np.float))

    return np.array(data)


def load_pkfile(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data_in = pickle.load(pickle_in)

    pickle_in.close()

    return data_in

def save_pkfile(outputfolder, data):

    pickle_out = open(outputfolder, "wb")

    pickle.dump(data, pickle_out)

    pickle_out.close()

    print(outputfolder, "saving successful !")

def norm(data):
    tmp_data = []
    for i, sample in enumerate(data):
        ver_sample = sample.transpose(1, 0)
        tmp_sample = []
        for row in ver_sample:
            # print(row)
            tmp_row = (row-np.amin(row))/(np.amax(row) - np.amin(row) + 0.0001)
            # print(tmp_row)
            tmp_sample.append(tmp_row)

        tmp_sample = np.array(tmp_sample)
        tmp_sample = tmp_sample.transpose(1, 0)
        tmp_data.append(tmp_sample)
    tmp_data = np.array(tmp_data)
    return tmp_data

def t_norm(data):
    result = []
    for row in data:
        result.append(tools.normalize(row)[0])
    result = np.array(result)
    return result

def transpose(raw_data):
    
    input_data = []

    for sample in raw_data:
        input_data.append( np.transpose(sample, (1, 0)) )

    return input_data

def resize(raw_data,length,ratio=1):
    input_data = np.zeros((len(raw_data),int(length*ratio),12))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data.shape[0],:data.shape[1]] = tools.normalize(data[0:int(length*ratio),:])['signal']

    input_data = np.transpose(input_data, (0, 2, 1))
    return input_data

def input_resizeing(raw_data,raw_labels,raw_labellist,raw_headers,ratio=1):
    input_data = np.zeros((len(raw_data),12,int(30000*ratio)))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data['val'].shape[0],:data['val'].shape[1]] = tools.normalize(data['val'][:,0:int(30000*ratio)])['signal']

    raw_labels = np.array(raw_labels)
    raw_labellist = np.array(raw_labellist)

    return input_data,raw_labels,raw_labellist


def get_length(raw_data):

    all_length = []
    
    for sample in raw_data:
        all_length.append(sample.shape[0])
    
    all_length = np.array(all_length)

    return all_length

import datetime
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
import editdistance
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def real_transformed_log():
    k = 21744

    # STEP 1: reading CSV and preprocessing
    path = '../datasets/real/real_transformed.csv'
    group_by = 'customer_id'  # CSV_COLUMN: Long running cases we would like to partition
    activity = 'activity'  # CSV_COLUMN: event
    time = 'time'  # CSV_COLUMN: timestamp column
    ground_truth = 'ticket_id'
    time_diff = 'time_diff'

    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    return preprocess(dataframe, k, group_by, activity, time, ground_truth, time_diff)


def synthetic_log(delay):
    k = 990

    path_base = '../datasets/synthetic/synthetic_1_'
    path = path_base + delay + '.csv'

    group_by = 'journey_id'
    activity = 'event'
    time = 'timestamp'
    time_diff = 'time_diff'

    # Read CSV
    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    dataframe.sort_values(by=[group_by, time], inplace=True)
    dataframe = dataframe[dataframe[time_diff].notna()]
    dataframe['new_guess_col'] = dataframe['last_trace']
    dataframe.drop(['last_trace'], axis=1, inplace=True)

    return segmentation(dataframe, k, activity, time_diff)


def leno_log(log_name):
    k = 50

    # STEP 1: reading CSV and preprocessing
    path = '../../leno/' + log_name + '_preprocessed.csv'
    group_by = 'userID'  # CSV_COLUMN: Long running cases we would like to partition
    activity = 'eventType'  # CSV_COLUMN: event
    time = 'timeStamp'  # CSV_COLUMN: timestamp column
    ground_truth = 'case_id'
    time_diff = 'time_diff'

    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    dataframe[time] = pd.to_datetime(dataframe[time])
    dataframe[time_diff] = dataframe.groupby(group_by)[time].shift(-1) - dataframe[time]
    dataframe[time_diff] = dataframe[time_diff].fillna(dataframe[time_diff].max())

    return preprocess(dataframe, k, group_by, activity, time, ground_truth, time_diff)


def preprocess(dataframe, k, group_by, activity, time, ground_truth, time_diff):
    dataframe.sort_values(by=[group_by, time], inplace=True)
    dataframe = dataframe[dataframe[time_diff].notna()]
    dataframe['next_guess_col'] = dataframe[ground_truth].shift(-1)
    dataframe['new_guess_col'] = (dataframe['next_guess_col'] != dataframe[ground_truth]) | (
        dataframe['next_guess_col'].isna())
    dataframe.drop(['next_guess_col'], axis=1, inplace=True)

    return segmentation(dataframe, k, activity, time_diff)

    # print(dataframe.head(100).to_string())


def segmentation(dataframe, k, activity, time_diff):
    df = pd.DataFrame()

    pred = dataframe.iloc[0]
    if isinstance(pred[time_diff], datetime.timedelta):
        pred[time_diff] = pred[time_diff].total_seconds()
    pred['TAP'] = pred[time_diff]
    if re.search('.*submit.*', pred[activity]):
        pred[time_diff] = pred[time_diff] * 40
    for f in tap_factors:
        pred['TOK_TAP_' + str(f)] = 0
        pred['TOK_TAP_' + str(f) + '_is_cut'] = False
    e_0 = pred[activity]
    x_0 = pred[time_diff]
    mean = x_0
    var = 1
    st_dev = 1
    varco = 0
    mptap_mean = 0
    mptap_var = 0
    mptap_st_dev = 1
    mptap_varco = 1
    pairs = {}
    all_times = []
    all_tap_times =[]
    mean_times = []
    mean_tap_times = []
    for i in range(1, len(dataframe)):
        if i % 500 == 0:
            print(i)
        current = dataframe.iloc[i]
        if isinstance(current[time_diff], datetime.timedelta):
            current[time_diff] = current[time_diff].total_seconds()
        current['TAP'] = current[time_diff]
        if re.search('.*submit.*', current[activity]):
            current[time_diff] = current[time_diff] * 40
        e_1 = current[activity]
        pair = e_0 + '_' + e_1
        for f in tap_factors:
            current['TOK_TAP_' + str(f) + '_is_cut'] = False
        for f in mptap_factors:
            pred['TOK_MPTAP_' + str(f) + '_is_cut'] = False
        x_1 = current[time_diff]
        if not math.isnan(x_1):
            if not math.isnan(x_0):
                if pair not in pairs:
                    pairs.update({pair: [1, x_0]})
                else:
                    pairs[pair][0] += 1
                    pair_mean = pairs[pair][1]
                    pair_mean_new = (x_0 + (pairs[pair][0] - 1) * pair_mean) / pairs[pair][0]
                    pairs[pair][1] = pair_mean_new

            for f in mptap_factors:
                pred['TOK_MPTAP_' + str(f)] = pairs[pair][1] - (mptap_mean + mptap_varco * f)
                if pred['TOK_MPTAP_' + str(f)] > 0:
                    pred['TOK_MPTAP_' + str(f) + '_is_cut'] = True

            all_times.append(pairs[pair][1])
            if len(all_times) > 100:
                all_times.pop(0)
            if pairs[pair][1] < np.quantile(all_times, 0.99):
                mean_times.append(pairs[pair][1])
            if mean_times:
                mptap_mean = np.mean(mean_times)
                mptap_var = np.var(mean_times)
            else:
                mptap_mean = pairs[pair][1]
                mptap_var = 1

            #mptap_mean_new = (pairs[pair][1] + i * mptap_mean) / (i + 1)
            #mptap_var = ((i - 1) * var + i * ((mptap_mean - mptap_mean_new) ** 2) + (
            #        (pairs[pair][1] - mptap_mean_new) ** 2)) / i
            mptap_st_dev = mptap_var ** 0.5
            #mptap_mean = mptap_mean_new
            mptap_varco = mptap_st_dev / mptap_mean

            for f in tap_factors:
                current['TOK_TAP_' + str(f)] = x_1 - (mean + varco * f)
                if current['TOK_TAP_' + str(f)] > 0:
                    current['TOK_TAP_' + str(f) + '_is_cut'] = True

            all_tap_times.append(x_1)
            if len(all_tap_times) > 100:
                all_tap_times.pop(0)
            if x_1 < np.quantile(all_tap_times, 0.99):
                mean_tap_times.append(x_1)
            if mean_tap_times:
                mean = np.mean(mean_tap_times)
                var = np.var(mean_tap_times)
            else:
                mean = x_1
                var = 1
            #mean_new = (x_1 + i * mean) / (i + 1)
            #var = ((i - 1) * var + i * ((mean - mean_new) ** 2) + ((x_1 - mean_new) ** 2)) / i
            st_dev = var ** 0.5
            #mean = mean_new
            varco = st_dev / mean

        df = pd.concat([df, pd.DataFrame([pred])], ignore_index=True)
        e_0 = e_1
        x_0 = x_1
        pred = current
    for f in mptap_factors:
        pred['TOK_MPTAP_' + str(f)] = pred[time_diff]
        pred['TOK_MPTAP_' + str(f) + '_is_cut'] = False
    df = pd.concat([df, pd.DataFrame([pred])], ignore_index=True)

    #print(df.head(10).to_string())
    #print(df.tail(10).to_string())
    print('mptap_mean', mptap_mean)
    print('mptap_std', mptap_st_dev)
    print('mptap_varco', mptap_varco)
    df = bernard_tap(df, k)
    df = bernard_lcpap(df, k, activity)
    df = df.reset_index()
    return df


# METHOD 1: TAP (using only the time to predict the case id)
# We simply insert a cut at the largest time difference
# and use a cumsum to assign a case_id
def bernard_tap(df, k):
    df['TAP_is_cut'] = False
    df.loc[df['TAP'].nlargest(k).index, 'TAP_is_cut'] = True
    df['TAP_discovered_case'] = df['TAP_is_cut'].shift(1).cumsum().fillna(0)
    return df


# METHOD 2: LCPAP (using the mean time between pairs of events)
# Same as method 1, but we replace the true time difference
# by the average time difference per pair of events
def bernard_lcpap(df, k, activity):
    df['next_activity'] = df[activity].shift(-1)
    df['pair'] = df[activity].astype(str) + '_' + df['next_activity'].astype(str)
    mapping = df.groupby('pair')['TAP'].mean()
    df['MPTAP'] = df['pair'].map(mapping)
    df['MPTAP_is_cut'] = False
    df.loc[df['MPTAP'].nlargest(k).index, 'MPTAP_is_cut'] = True
    df['MPTAP_discovered_case'] = df['MPTAP_is_cut'].shift(1).cumsum().fillna(0)
    return df


def get_all_uis(df):
    all_uis = []
    for i in range(len(df)):
        all_uis.append(df.iloc[i][0])
    return all_uis


def find_partitions(df, label):
    partition = []
    section = []
    for i in range(len(df)):
        section.append(df.iloc[i][0])
        if df.iloc[i][label]:
            partition.append(section)
            section = []
    return partition


def get_edit_distance(df, label):
    edit_distances = []
    all_uis = get_all_uis(df)
    discovered_segments = find_partitions(df, label + '_is_cut')
    true_segments = find_partitions(df, 'new_guess_col')

    for discovered_seg in discovered_segments:
        covered_traces = []
        for true_seg in true_segments:
            if any([i in discovered_seg for i in true_seg]) and len(true_seg) > 0:
                covered_traces.append(true_seg)

        edit_distance = 1
        min_seg = []
        for true_seg in covered_traces:
            dist = (editdistance.eval([all_uis[i] for i in discovered_seg if i < len(all_uis)],
                                      [all_uis[i] for i in true_seg if i < len(all_uis)]) / max(len(discovered_seg),
                                                                                                len(true_seg)))
            if dist < edit_distance:
                edit_distance = dist
                min_seg = true_seg
        if len(min_seg) > 0:
            edit_distances.append(edit_distance)
    mean_edit_distance = np.mean(edit_distances)
    return mean_edit_distance


def get_statistics(df, label, w):
    print(label)

    traces = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(w, len(df)):
        if df.iloc[i]['new_guess_col'] == df.iloc[i][label + '_is_cut']:
            if df.iloc[i]['new_guess_col']:
                traces += 1
                tp += 1
            else:
                tn += 1
        else:
            if df.iloc[i]['new_guess_col']:
                traces += 1
                fn += 1
            else:
                fp += 1

    try:
        tpr_recall = tp / (tp + fn)
    except ZeroDivisionError:
        tpr_recall = 0

    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = 0

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        fscore = 2 * precision * tpr_recall / (precision + tpr_recall)
    except ZeroDivisionError:
        fscore = 0

    return traces, tp, fp, fn, tn, precision, tpr_recall, fpr, fscore


def evaluate(logs):
    with pd.ExcelWriter('./results/results_complete/lcpap_res_' + parameter + '.xlsx') as mptap_writer:
        for log_name in logs:
            df = logs[log_name]
            results = pd.DataFrame()
            results.index = ['traces', 'tp', 'fp', 'fn', 'tn', 'prec', 'tpr_recall', 'fpr', 'auc', 'fscore', 'med']

            auc = generate_roc(df, log_name)

            for w in warm_ups:
                label = 'MPTAP'
                med_mptap = get_edit_distance(df.iloc[w:], label)
                traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                results.insert(0, 'bern_wu_' + str(w),
                               [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_mptap], True)
                for f in mptap_factors:
                    label = 'TOK_MPTAP_' + str(f)
                    med_tok = get_edit_distance(df.iloc[w:], label)
                    traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                    results.insert(0, 'fmptap_' + str(f) + '_wu_' + str(w),
                                   [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tok], True)

            print(results.to_string())
            results.to_excel(mptap_writer, sheet_name=log_name)
        # mptap_writer.save()

    with pd.ExcelWriter('./results/results_complete/tap_res_' + parameter + '.xlsx') as tap_writer:
        for log_name in logs:
            df = logs[log_name]
            results = pd.DataFrame()
            results.index = ['traces', 'tp', 'fp', 'fn', 'tn', 'prec', 'tpr_recall', 'fpr', 'auc', 'fscore', 'med']

            for w in warm_ups:
                label = 'TAP'
                med_tap = get_edit_distance(df.iloc[w:], label)
                traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                results.insert(0, 'bern_wu_' + str(w),
                               [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tap], True)
                for f in tap_factors:
                    label = 'TOK_TAP_' + str(f)
                    med_tok = get_edit_distance(df.iloc[w:], label)
                    traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                    results.insert(0, 'ftap_' + str(f) + '_wu_' + str(w),
                                   [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tok], True)

            print(results.to_string())
            results.to_excel(tap_writer, sheet_name=log_name)
        # tap_writer.save()


def generate_roc(df, log_name):
    fpr = {}
    tpr = {}
    thresholds = {}
    auc = {}
    labels = ['TAP', 'MPTAP']
    for f in tap_factors:
        labels.append('TOK_TAP_' + str(f))
        labels.append('TOK_TAP_' + str(f) + '_is_cut')
    for f in mptap_factors:
        labels.append('TOK_MPTAP_' + str(f))
        labels.append('TOK_MPTAP_' + str(f) + '_is_cut')
    for c in labels:
        fpr[c], tpr[c], thresholds[c] = roc_curve(df['new_guess_col'], df[c])
        auc[c] = roc_auc_score(df['new_guess_col'], df[c])
        # print(fpr[c], tpr[c], thresholds[c])
        # print('auc', c, auc[c])
    plt.plot(fpr['TAP'], tpr['TAP'], label='TAP')
    plt.plot(fpr['MPTAP'], tpr['MPTAP'], label='LCPAP')
    plt.plot(fpr['TOK_TAP_1'], tpr['TOK_TAP_1'], label='Streaming TAP')
    plt.plot(fpr['TOK_MPTAP_1'], tpr['TOK_MPTAP_1'], label='Streaming LCPAP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig('results/results_complete/roc_curves/' + log_name + '_roc_' + parameter + '.eps', format='eps')
    plt.close()
    return auc


segmented_logs = {}
tap_factors = [1]
mptap_factors = [1]
warm_ups = [0]  # , 100, 500, 1000, 2000]
all_delays = ['delay1.0', 'delay0.5', 'delay0.45', 'delay0.35']
parameter = 'quant_0.99_ui'

segmented_logs['reimb'] = leno_log('Reimbursement')
segmented_logs['student'] = leno_log('StudentRecord')
# segmented_logs['real'] = real_transformed_log()

# for d in all_delays:
#     segmented_logs[d] = synthetic_log(d)

evaluate(segmented_logs)
for log_name in segmented_logs:
    df = segmented_logs[log_name]
    auc = generate_roc(df, log_name)
    print(log_name, auc)
import datetime
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
import editdistance
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# method to process the real-life customer journey log from Bernard et al.
def real_transformed_log():
    k = 21744

    # STEP 1: reading CSV and preprocessing
    path = '../../datasets/real/real_transformed.csv'
    group_by = 'customer_id'  # CSV_COLUMN: Long running cases we would like to partition
    activity = 'activity'  # CSV_COLUMN: event
    time_stamp = 'time'  # CSV_COLUMN: timestamp column
    ground_truth = 'ticket_id'
    time_diff = 'time_diff'

    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    dataframe = dataframe[dataframe[time_diff].notna()]
    return preprocess(dataframe, k, group_by, activity, time_stamp, ground_truth)


# method to process the synthetic UI logs by Bernard et al.
def synthetic_log(delay):
    k = 990

    path_base = '../../datasets/synthetic/synthetic_1_'
    path = path_base + delay + '.csv'

    group_by = 'journey_id'
    activity = 'event'
    time_stamp = 'timestamp'
    time_diff = 'time_diff'

    # Read CSV
    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    dataframe.sort_values(by=[group_by, time_stamp], inplace=True)
    dataframe = dataframe[dataframe[time_diff].notna()]
    dataframe['new_guess_col'] = dataframe['last_trace']
    dataframe.drop(['last_trace'], axis=1, inplace=True)

    return segmentation(dataframe, k, activity, time_stamp)


# method to process the real-life UI logs from Leno et al.
def leno_log(log_name):
    k = 50

    # STEP 1: reading CSV and preprocessing
    path = '../../datasets/leno/' + log_name + '.csv'
    group_by = 'timeStamp'  # CSV_COLUMN: Long running cases we would like to partition
    activity = 'eventType'  # CSV_COLUMN: event
    time_stamp = 'timeStamp'  # CSV_COLUMN: timestamp column
    ground_truth = 'case_id'

    dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
    dataframe[time_stamp] = pd.to_datetime(dataframe[time_stamp], format='%Y-%m-%dT%H:%M:%S.%fZ')
    dataframe = dataframe.sort_values(by=[time_stamp], ignore_index=True)
    dataframe[ground_truth] = 0
    case_id = 0

    # manually create the ground truth according to Leno et al.
    for i in range(0, len(dataframe)):
        if dataframe['target.innerText'][i] in ['Add another response', 'Add another response.']:
            case_id += 1
        dataframe[ground_truth][i] = case_id

    return preprocess(dataframe, k, group_by, activity, time_stamp, ground_truth)


# method to filter the stream from redundant events
def filtering(row1, row2, activity):

    def mergeNavigationCellCopy(r1, r2, activity):

        # merge getCell or editField and copy events
        if r1[activity] == "getCell" or r1[activity] == "editCell":
            r1['timeStamp'] = r2['timeStamp']
            r1['content'] = r2['content']
            r1[activity] = "copyCell"

        # merge getRange and copy events
        elif r1[activity] == "getRange":
            r1['timeStamp'] = r2['timeStamp']
            r1['content'] = r2['content']
            r1[activity] = "copyRange"

        return r1

    event1 = row1[activity]
    if event1 == 'copy' or event1 == 'clickTextField' or event1 == 'form_submit' or event1 == 'ignore':
        return None
    elif row2[activity] == 'copy':
        row = mergeNavigationCellCopy(row1, row2, activity)
    else:
        row = row1

    if row[activity] == "clickButton" and pd.notnull(row['target.type']):
        row[activity] = str(row[activity]) + '[' + str(row['target.type']) + ']'

    return row


# method to sort the log by its timestamp and retrieve the ground truth
def preprocess(dataframe, k, group_by, activity, time_stamp, ground_truth):
    dataframe.sort_values(by=[group_by, time_stamp], inplace=True)
    dataframe = dataframe[dataframe[time_stamp].notna()]
    dataframe['next_guess_col'] = dataframe[ground_truth].shift(-1)
    dataframe['new_guess_col'] = (dataframe['next_guess_col'] != dataframe[ground_truth]) | (
        dataframe['next_guess_col'].isna())
    dataframe.drop(['next_guess_col'], axis=1, inplace=True)

    return segmentation(dataframe, k, activity, time_stamp)


# method for segmenting the stream
def segmentation(dataframe, k, activity, time_stamp):
    df = pd.DataFrame()
    time_diff = 'time_diff'

    t = time.time()

    j = 0
    pred = dataframe.iloc[j]
    j += 1
    succ = dataframe.iloc[j]

    if time_diff not in pred:
        pred[time_diff] = succ[time_stamp] - pred[time_stamp]
    if isinstance(pred[time_diff], datetime.timedelta):
        pred[time_diff] = pred[time_diff].total_seconds()

    pred['TAP'] = pred[time_diff]

    if any(ele in pred[activity] for ele in keyword_set):
        pred[time_diff] = pred[time_diff] * f

    pred['TOK_TAP'] = 0
    pred['TOK_TAP_is_cut'] = False

    e_0 = pred[activity]
    x_0 = pred[time_diff]
    mean = x_0
    var = 1
    st_dev = 1
    varco = 0
    mptap_mean = x_0
    mptap_var = 0
    mptap_varco = 1
    pairs = {}
    processing_time_avg = (time.time() - t) * 1000

    for i in range(j, len(dataframe) - 1):
        if i % 1000 == 0:
            print(i)

        t = time.time()

        current = dataframe.iloc[i]

        if time_diff not in pred:
            pred[time_diff] = current[time_stamp] - pred[time_stamp]
        if isinstance(pred[time_diff], datetime.timedelta):
            pred[time_diff] = pred[time_diff].total_seconds()

        pred['TAP'] = pred[time_diff]

        if any(ele in pred[activity] for ele in keyword_set):
            pred[time_diff] = pred[time_diff] * f

        x_0 = pred[time_diff]
        e_1 = current[activity]
        pair = e_0 + '_' + e_1

        pred['TOK_TAP_is_cut'] = False
        pred['TOK_MPTAP_is_cut'] = False

        if not math.isnan(x_0):
            if pair not in pairs:
                pairs.update({pair: [1, x_0]})
            else:
                pairs[pair][0] += 1
                pair_mean = pairs[pair][1]
                pair_mean_new = (x_0 + (pairs[pair][0] - 1) * pair_mean) / pairs[pair][0]
                pairs[pair][1] = pair_mean_new

            pred['TOK_MPTAP'] = pairs[pair][1] - (mptap_mean + mptap_varco)
            pred['TOK_MPTAP_is_cut'] = pred['TOK_MPTAP'] > 0

            mptap_mean_new = (pairs[pair][1] + i * mptap_mean) / (i + 1)
            mptap_var = ((i - 1) * mptap_var + i * ((mptap_mean - mptap_mean_new) ** 2) + (
                    (pairs[pair][1] - mptap_mean_new) ** 2)) / i
            mptap_st_dev = mptap_var ** 0.5
            mptap_mean = mptap_mean_new
            mptap_varco = mptap_st_dev / mptap_mean

            processing_time = (time.time() - t) * 1000

            pred['TOK_TAP'] = x_0 - (mean + varco)
            pred['TOK_TAP_is_cut'] = pred['TOK_TAP'] > 0

            mean_new = (x_0 + i * mean) / (i + 1)
            var = ((i - 1) * var + i * ((mean - mean_new) ** 2) + ((x_0 - mean_new) ** 2)) / i
            st_dev = var ** 0.5
            mean = mean_new
            varco = st_dev / mean

        df = pd.concat([df, pd.DataFrame([pred])], ignore_index=True)
        e_0 = e_1
        pred = current

        processing_time_avg = (processing_time + i * processing_time_avg) / (i + 1)

    second_last = dataframe.iloc[len(dataframe) - 2]
    last = dataframe.iloc[len(dataframe) - 1]

    second_last = filtering(second_last, last, activity)

    if second_last is not None:
        if time_diff not in second_last:
            second_last[time_diff] = last[time_stamp] - second_last[time_stamp]
        if isinstance(second_last[time_diff], datetime.timedelta):
            second_last[time_diff] = second_last[time_diff].total_seconds()
        second_last['TAP'] = second_last[time_diff]

        second_last['TOK_TAP'] = second_last[time_diff] - (mean + varco)
        second_last['TOK_TAP_is_cut'] = second_last['TOK_TAP'] > 0

        second_last['TOK_MPTAP'] = second_last[time_diff] - (mptap_mean + mptap_varco)
        second_last['TOK_MPTAP_is_cut'] = second_last['TOK_MPTAP'] > 0

        df = pd.concat([df, pd.DataFrame([second_last])], ignore_index=True)

    event1 = last[activity]
    if event1 == 'copy' or event1 == 'clickTextField' or event1 == 'form_submit' or event1 == 'ignore':
        last = None

    if last is not None:
        last[time_diff] = 1000000000
        if 'time_diff_norm' in last:
            last['time_diff_norm'] = 1000000000
        last['TAP'] = last[time_diff]

        last['TOK_TAP'] = last[time_diff]
        last['TOK_TAP_is_cut'] = True

        last['TOK_MPTAP'] = last[time_diff]
        last['TOK_MPTAP_is_cut'] = True

        df = pd.concat([df, pd.DataFrame([last])], ignore_index=True)
    else:
        df['TOK_TAP'][-1] = 1000000000
        df['TOK_TAP_is_cut'][-1] = True

        df['TOK_MPTAP'][-1] = 1000000000
        df['TOK_MPTAP_is_cut'][-1] = True

    print('mean', mean)
    print('std', st_dev)
    print('varco', varco)
    print('number of pairs', len(pairs))
    print('avg processing time', processing_time_avg)
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


# method to support the computation of the edit distance
def get_all_uis(df):
    all_uis = []
    for i in range(len(df)):
        all_uis.append(df.iloc[i][0])
    return all_uis


# method to support the computation of the edit distance
def find_partitions(df, label):
    partition = []
    section = []
    for i in range(len(df)):
        section.append(df.iloc[i][0])
        if df.iloc[i][label]:
            partition.append(section)
            section = []
    return partition


# method to compute the edit distance for the evaluation
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


# method to compute the evaluation metrics such as TPR, FPR and F1 score
def get_statistics(df, label, w):
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


# method to evaluate the results and save them into an Excel sheet
def evaluate(logs):
    with pd.ExcelWriter('../results/additional_results/' + parameter + '.xlsx') as mptap_writer:
        for log_name in logs:
            df = logs[log_name]
            results = pd.DataFrame()
            results.index = ['traces', 'tp', 'fp', 'fn', 'tn', 'prec', 'tpr_recall', 'fpr', 'auc', 'fscore', 'med']

            # df.to_csv('./results/results_complete/' + log_name + '_segmented_' + parameter + '.csv', sep=',', index=False)
            auc = generate_roc(df, log_name)

            for w in warm_ups:
                label = 'TOK_MPTAP'
                med_tok = get_edit_distance(df.iloc[w:], label)
                traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                results.insert(0, 'TOK_wu_' + str(w),
                               [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tok], True)

            print(results.to_string())
            results.to_excel(mptap_writer, sheet_name=log_name)
        # mptap_writer.save()

    '''
    with pd.ExcelWriter('./results/results_complete/tap_res_' + parameter + '.xlsx') as tap_writer:
        for log_name in logs:
            df = logs[log_name]
            results = pd.DataFrame()
            results.index = ['traces', 'tp', 'fp', 'fn', 'tn', 'prec', 'tpr_recall', 'fpr', 'auc', 'fscore', 'med']

            for w in warm_ups:
                label = 'TAP'
                med_tap = get_edit_distance(df.iloc[w:], label)
                traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                results.insert(0, 'Bernard_wu_' + str(w),
                               [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tap], True)
                label = 'TOK_TAP'
                med_tok = get_edit_distance(df.iloc[w:], label)
                traces, tp, fp, fn, tn, precision, tpr, fpr, fscore = get_statistics(df, label, w)
                results.insert(0, 'TOK_wu_' + str(w),
                               [traces, tp, fp, fn, tn, precision, tpr, fpr, auc[label], fscore, med_tok], True)

            print(results.to_string())
            results.to_excel(tap_writer, sheet_name=log_name)
        # tap_writer.save()
    '''


# method to generate the ROC and AUC score for the evaluation and save them as an eps file
def generate_roc(df, log_name):
    fpr = {}
    tpr = {}
    thresholds = {}
    auc = {}
    labels = ['TAP', 'MPTAP', 'TOK_TAP', 'TOK_MPTAP']
    for c in labels:
        fpr[c], tpr[c], thresholds[c] = roc_curve(df['new_guess_col'], df[c])
        auc[c] = roc_auc_score(df['new_guess_col'], df[c])
    plt.plot(fpr['TAP'], tpr['TAP'], label='TAP (AUC: ' + str(round(auc['TAP'], 2)) + ')')
    plt.plot(fpr['MPTAP'], tpr['MPTAP'], label='LCPAP (AUC: ' + str(round(auc['MPTAP'], 2)) + ')')
    plt.plot(fpr['TOK_TAP'], tpr['TOK_TAP'], label='Streaming TAP (AUC: ' + str(round(auc['TOK_TAP'], 2)) + ')')
    plt.plot(fpr['TOK_MPTAP'], tpr['TOK_MPTAP'], label='Our approach (AUC: ' + str(round(auc['TOK_MPTAP'], 2)) + ')')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    # plt.savefig('results/results_complete/roc_curves/' + log_name + '_roc_' + parameter + '.eps', format='eps')
    plt.close()
    return auc


# keyword set for keywords out of the IBM guidelines (taken from Rebmann, van der Aa)
keyword_set = ["submit", "save", "ok", "confirm", "apply", "add", "cancel", "close", "delete", "done", "download",
               "finish", "next", "ok", "post", "reject", "send", "update", "upload", "fertig", "speichern", "anwenden",
               "bernehmen"]


segmented_logs = {}
warm_ups = [0]
f = 40
all_delays = ['delay1.0', 'delay0.5']
parameter = 'no_filtering'

segmented_logs['reimb'] = leno_log('Reimbursement')
segmented_logs['student'] = leno_log('StudentRecord')
# segmented_logs['real'] = real_transformed_log()

# for d in all_delays:
#     segmented_logs[d] = synthetic_log(d)

evaluate(segmented_logs)

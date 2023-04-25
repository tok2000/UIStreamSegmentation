import math
import pandas as pd
import numpy as np
import warnings
import editdistance

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This is a very simple example to describe how customer journey partitioning can be performed using the
3 techniques described in the paper. Let's say we would like to partition the event logs synthetic_1_delay2.0.csv
which contains 10 customer journeys into 1000 distinct event logs
'''

k = 21744
tap_factors = [0.5, 1]
mptap_factors = [0.5, 1]
warm_ups = [0]  # , 100, 500, 1000, 5000]

# STEP 1: reading CSV and preprocessing
path = '../datasets/real/real_transformed.csv'

group_by = 'customer_id'  # CSV_COLUMN: Long running cases we would like to partition
activity = 'activity'  # CSV_COLUMN: event
time = 'time'  # CSV_COLUMN: timestamp column
ground_truth = 'ticket_id'
time_diff = 'time_diff'

# Read CSV
dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
dataframe.sort_values(by=[group_by, time], inplace=True)
dataframe = dataframe[dataframe[time_diff].notna()]
dataframe['next_guess_col'] = dataframe[ground_truth].shift(-1)
dataframe['new_guess_col'] = (dataframe['next_guess_col'] != dataframe[ground_truth]) | (
    dataframe['next_guess_col'].isna())
dataframe.drop(['next_guess_col'], axis=1, inplace=True)

# print(dataframe.head(100).to_string())


df = pd.DataFrame()

pred = dataframe.iloc[0]
for f in tap_factors:
    pred['TOK_TAP_' + str(f) + '_is_cut'] = False
e_0 = pred[activity]
x_0 = pred[time_diff]
mean = x_0
mptap_mean = 0
var = 0
mptap_var = 0
st_dev = 0
mptap_st_dev = 0
varco = 0
mptap_varco = 0
pairs = {}
for i in range(1, len(dataframe)):
    if i % 500 == 0:
        print(i)
    current = dataframe.iloc[i]
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
                # pair_var = ((pairs[pair][0] - 2) * pairs[pair][2] + (pairs[pair][0] - 1) *
                #            ((pair_mean - pair_mean_new) ** 2) + ((x_0 - pair_mean_new) ** 2)) / (pairs[pair][0] - 1)
                # pair_st_dev = pair_var ** 0.5
                # pairs[pair][2] = pair_var
        pred['TOK_MPTAP'] = pairs[pair][1]

        for f in mptap_factors:
            if pairs[pair][1] > mean + varco * f:
                pred['TOK_MPTAP_' + str(f) + '_is_cut'] = True

        mptap_mean_new = (pairs[pair][1] + i * mptap_mean) / (i + 1)
        mptap_var = ((i - 1) * var + i * ((mptap_mean - mptap_mean_new) ** 2) + (
                (pairs[pair][1] - mptap_mean_new) ** 2)) / i
        mptap_st_dev = mptap_var ** 0.5
        mptap_mean = mptap_mean_new
        mptap_varco = mptap_st_dev / mptap_mean

        for f in tap_factors:
            if x_1 > mean + varco * f:
                current['TOK_TAP_' + str(f) + '_is_cut'] = True

        mean_new = (x_1 + i * mean) / (i + 1)
        var = ((i - 1) * var + i * ((mean - mean_new) ** 2) + ((x_1 - mean_new) ** 2)) / i
        st_dev = var ** 0.5
        mean = mean_new
        varco = st_dev / mean
        # print(i)
        # print('st_dev: ' + str(st_dev))
        # print('varco: ' + str(varco))
        # print('sen: ' + str(sen))

    df = df.append(pred)
    e_0 = e_1
    x_0 = x_1
    pred = current
df = df.append(current)

print(df.head(100).to_string())

# METHOD 1: TAP (using only the time to predict the case id)
# We simply insert a cut at the largest time difference
# and use a cumsum to assign a case_id
df['TAP_is_cut'] = False
df.loc[df[time_diff].nlargest(k).index, 'TAP_is_cut'] = True
df['TAP_discovered_case'] = df['TAP_is_cut'].shift(1).cumsum().fillna(0)

# METHOD 2: LCPAP (using the mean time between pairs of events)
# Same as method 1, but we replace the true time difference
# by the average time difference per pair of events
df['next_activity'] = df[activity].shift(-1)
df['pair'] = df[activity].astype(str) + '_' + df['next_activity'].astype(str)
mapping = df.groupby('pair')[time_diff].mean()
df['MPTAP'] = df['pair'].map(mapping)
df['MPTAP_is_cut'] = False
df.loc[df['MPTAP'].nlargest(k).index, 'MPTAP_is_cut'] = True
df['MPTAP_discovered_case'] = df['MPTAP_is_cut'].shift(1).cumsum().fillna(0)


def get_all_uis(df):
    all_uis = []
    for i in range(len(df)):
        all_uis.append(df.iloc[i][0])
    # print(all_uis)
    return all_uis


def find_partitions(df, label):
    partition = []
    section = []
    for i in range(len(df)):
        section.append(df.iloc[i][0])
        if df.iloc[i][label]:
            partition.append(section)
            section = []
    # print(label + str(partition))
    return partition


def get_edit_distance(df, label):
    edit_distances = []
    all_uis = get_all_uis(df)
    discovered_segments = find_partitions(df, label)
    true_segments = find_partitions(df, 'new_guess_col')
    # print(discovered_segments)
    # print(true_segments)

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
        # print("disc", discovered_seg)
        # print("min_true", min_seg)
        if len(min_seg) > 0:
            # print(edit_distance)
            edit_distances.append(edit_distance)
    mean_edit_distance = np.mean(edit_distances)
    return mean_edit_distance


def get_statistics(corr, wrong, traces):
    precision = corr / (corr + wrong)
    recall = corr / traces
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore


def get_metrics(df, results, f, w, label):
    print(label + ': f: ' + str(f) + ', w: ' + str(w))

    bern_corr = 0
    bern_wrong = 0
    tok_corr = 0
    tok_wrong = 0
    traces = 0
    for i in range(w, len(df)):
        if df.iloc[i]['new_guess_col']:
            traces += 1
            if df.iloc[i][label + '_is_cut']:
                bern_corr += 1
            if df.iloc[i]['TOK_' + label + '_' + str(f) + '_is_cut']:
                tok_corr += 1
        else:
            if df.iloc[i][label + '_is_cut']:
                bern_wrong += 1
            if df.iloc[i]['TOK_' + label + '_' + str(f) + '_is_cut']:
                tok_wrong += 1

    prec_bern, rec_bern, f_bern = get_statistics(bern_corr, bern_wrong, traces)
    prec_tok, rec_tok, f_tok = get_statistics(tok_corr, tok_wrong, traces)

    results.insert(0, label + '_' + str(f) + '_wu_' + str(w),
                   [traces, bern_corr, bern_wrong, prec_bern, rec_bern, f_bern, tok_corr, tok_wrong, prec_tok, rec_tok,
                    f_tok], True)
    return results


def evaluate(df):
    results = pd.DataFrame()
    results.index = ['traces', 'tap_corr', 'tap_wrong', 'prec_tap', 'recall_tap', 'fscore_tap', 'tok_corr',
                     'tok_wrong', 'prec_tok', 'recall_tok', 'fscore_tok']

    for f in tap_factors:
        for w in warm_ups:
            results = get_metrics(df, results, f, w, 'TAP')
        print('tok_tap_ed', get_edit_distance(df, 'TOK_TAP_' + str(f) + '_is_cut'))

    print('bernard_tap_ed', get_edit_distance(df, 'TAP_is_cut'))

    print(results.to_string())
    results.to_excel('./results/tap_res_new1.xlsx')

    results = pd.DataFrame()
    results.index = ['traces', 'mptap_corr', 'mptap_wrong', 'prec_mptap', 'recall_mptap', 'fscore_mptap', 'tok_corr',
                     'tok_wrong', 'prec_tok', 'recall_tok', 'fscore_tok']

    for f in mptap_factors:
        for w in warm_ups:
            results = get_metrics(df, results, f, w, 'MPTAP')
        print('tok_mptap_ed', get_edit_distance(df, 'TOK_MPTAP_' + str(f) + '_is_cut'))

    print('bernard_mptap_ed', get_edit_distance(df, 'MPTAP_is_cut'))

    print(results.to_string())
    results.to_excel('./results/mptap_res_new1.xlsx')


df = df.reset_index()
evaluate(df)

import math
import pandas as pd
import warnings
import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This is a very simple example to describe how customer journey partitioning can be performed using the
3 techniques described in the paper. Let's say we would like to partition the event logs synthetic_1_delay2.0.csv
which contains 10 customer journeys into 1000 distinct event logs
'''

k = 50
tap_factors = [-10, -5, -1, 0, 0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10]
mptap_factors = [-10, -5, -1, 0, 0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10]
warm_ups = [0]  # , 100, 500, 1000, 5000]

# STEP 1: reading CSV and preprocessing
log_name = 'Reimbursement'
path = '../../leno/' + log_name + '_preprocessed.csv'

group_by = 'userID'  # CSV_COLUMN: Long running cases we would like to partition
activity = 'eventType'  # CSV_COLUMN: event
time = 'timeStamp'  # CSV_COLUMN: timestamp column
ground_truth = 'case_id'
time_diff = 'time_diff'

# Read CSV
dataframe = pd.read_csv(path, nrows=None, dtype={group_by: str, activity: str})
dataframe[time] = pd.to_datetime(dataframe[time])
dataframe.sort_values(by=[group_by, time], inplace=True)
dataframe[time_diff] = dataframe.groupby(group_by)[time].shift(-1) - dataframe[time]
dataframe[time_diff] = dataframe['time_diff'].fillna(dataframe['time_diff'].max())
dataframe = dataframe[dataframe[time_diff].notna()]
dataframe['next_guess_col'] = dataframe[ground_truth].shift(-1)
dataframe['new_guess_col'] = (dataframe['next_guess_col'] != dataframe[ground_truth]) | (
    dataframe['next_guess_col'].isna())
dataframe.drop(['next_guess_col'], axis=1, inplace=True)

# print(dataframe.to_string())
# dataframe.loc[56, time_diff] = dataframe.loc[2215, time_diff]
# dataframe.loc[1040, time_diff] = 7

df = pd.DataFrame()

pred = dataframe.iloc[0]
pred[time_diff] = pred[time_diff].total_seconds()
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
pairs = {}
for i in range(1, len(dataframe)):
    if i % 500 == 0:
        print(i)
    current = dataframe.iloc[i]
    current[time_diff] = current[time_diff].total_seconds()
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
        pred['pair_2'] = pair + '_' + str(pairs[pair][1])

        for f in mptap_factors:
            if pairs[pair][1] > mean + st_dev * f:
                pred['TOK_MPTAP_' + str(f) + '_is_cut'] = True

        mptap_mean_new = (pairs[pair][1] + i * mptap_mean) / (i + 1)
        mptap_var = ((i - 1) * var + i * ((mptap_mean - mptap_mean_new) ** 2) + (
                    (pairs[pair][1] - mptap_mean_new) ** 2)) / i
        mptap_st_dev = mptap_var ** 0.5
        mptap_mean = mptap_mean_new

        for f in tap_factors:
            if x_1 > mean + st_dev * f:
                current['TOK_TAP_' + str(f) + '_is_cut'] = True

        mean_new = (x_1 + i * mean) / (i + 1)
        var = ((i - 1) * var + i * ((mean - mean_new) ** 2) + ((x_1 - mean_new) ** 2)) / i
        st_dev = var ** 0.5
        mean = mean_new

    df = df.append(pred)
    e_0 = e_1
    x_0 = x_1
    pred = current
df = df.append(current)

# METHOD 1: TAP (using only the time to predict the case id)
# We simply insert a cut at the largest time difference
# and use a cumsum to assign a case_id
df['TAP_is_cut'] = False
df.loc[df[time_diff].nlargest(k).index, 'TAP_is_cut'] = True
df['TAP_discovered_case'] = df['TAP_is_cut'].shift(1).cumsum().fillna(0)

results_tap = pd.DataFrame()
results_tap.index = ['tap', 'tok', 'both', 'tap_corr', 'tok_corr', 'traces', 'tap_wrong', 'tok_wrong']

for f in tap_factors:
    for w in warm_ups:
        print('tap: f: ' + str(f) + ', w: ' + str(w))
        tap = 0
        tok = 0
        both = 0
        for i in range(w, len(df)):
            if df.iloc[i]['TAP_is_cut']:
                if df.iloc[i]['TOK_TAP_' + str(f) + '_is_cut']:
                    both += 1
                else:
                    tap += 1
            else:
                if df.iloc[i]['TOK_TAP_' + str(f) + '_is_cut']:
                    tok += 1

        tap_corr = 0
        tap_wrong = 0
        tok_corr = 0
        tok_wrong = 0
        traces = 0
        for i in range(w, len(df)):
            if df.iloc[i]['new_guess_col']:
                traces += 1
                if df.iloc[i]['TAP_is_cut']:
                    tap_corr += 1
                if df.iloc[i]['TOK_TAP_' + str(f) + '_is_cut']:
                    tok_corr += 1
            else:
                if df.iloc[i]['TAP_is_cut']:
                    tap_wrong += 1
                if df.iloc[i]['TOK_TAP_' + str(f) + '_is_cut']:
                    tok_wrong += 1

        results_tap.insert(0, 'ftap_' + str(f) + '_wu_' + str(w),
                           [tap, tok, both, tap_corr, tok_corr, traces, tap_wrong, tok_wrong], True)

print(results_tap.to_string())
results_tap.to_excel('./results/tap_res_leno.xlsx')

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

print(df.head(400).to_string(columns=['case_id', 'eventType', 'target.innerText', time_diff, 'new_guess_col', 'MPTAP', 'TOK_MPTAP', 'pair_2']))

results_mptap = pd.DataFrame()
results_mptap.index = ['mptap', 'tok', 'both', 'mptap_corr', 'tok_corr', 'traces', 'mptap_wrong', 'tok_wrong']

for f in mptap_factors:
    for w in warm_ups:
        print('mptap: f: ' + str(f) + ', w: ' + str(w))
        mptap = 0
        tok = 0
        both = 0
        for i in range(w, len(df)):
            if df.iloc[i]['MPTAP_is_cut']:
                if df.iloc[i]['TOK_MPTAP_' + str(f) + '_is_cut']:
                    both += 1
                else:
                    mptap += 1
            else:
                if df.iloc[i]['TOK_MPTAP_' + str(f) + '_is_cut']:
                    tok += 1

        mptap_corr = 0
        mptap_wrong = 0
        tok_corr = 0
        tok_wrong = 0
        traces = 0
        for i in range(w, len(df)):
            if df.iloc[i]['new_guess_col']:
                traces += 1
                if df.iloc[i]['MPTAP_is_cut']:
                    mptap_corr += 1
                if df.iloc[i]['TOK_MPTAP_' + str(f) + '_is_cut']:
                    tok_corr += 1
            else:
                if df.iloc[i]['MPTAP_is_cut']:
                    mptap_wrong += 1
                if df.iloc[i]['TOK_MPTAP_' + str(f) + '_is_cut']:
                    tok_wrong += 1
        results_mptap.insert(0, 'fmptap_' + str(f) + '_wu_' + str(w),
                             [mptap, tok, both, mptap_corr, tok_corr, traces, mptap_wrong, tok_wrong], True)

# print(df.head(100).to_string())
# print(pairs)

print(results_mptap.to_string())
results_mptap.to_excel('./results/mptap_res_leno.xlsx')

import math
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This is a very simple example to describe how customer journey partitioning can be performed using the
3 techniques described in the paper. Let's say we would like to partition the event logs synthetic_1_delay2.0.csv
which contains 10 customer journeys into 1000 distinct event logs
'''
k = 1000

# STEP 1: reading CSV and preprocessing
path = '../datasets/synthetic/synthetic_1_delay0.1.csv'

group_by = 'journey_id' # CSV_COLUMN: Long running cases we would like to partition
activity = 'event'      # CSV_COLUMN: event
time = 'timestamp'      # CSV_COLUMN: timestamp column

# Read CSV
f = pd.read_csv(path, nrows=None, dtype={group_by:str, activity:str})
f.sort_values(by=[group_by, time], inplace=True)

df = pd.DataFrame()

pred = f.iloc[0]
pred['TOK_TAP_is_cut'] = False
e_0 = pred[activity]
x_0 = pred['time_diff']
mean = x_0
mptap_mean = 0
var = 0
mptap_var = 0
st_dev = 0
mptap_st_dev = 0
c = 0
pairs = {}
for i in range(1, len(f)):
    if i % 500 == 0:
        print(i)
    current = f.iloc[i]
    e_1 = current[activity]
    pair = e_0 + '_' + e_1
    current['TOK_TAP_is_cut'] = False
    pred['TOK_MPTAP_is_cut'] = False
    x_1 = current['time_diff']
    if not math.isnan(x_1):
        if not math.isnan(x_0):
            if pair not in pairs:
                pairs.update({pair : [1, x_0]})
            else:
                pairs[pair][0] += 1
                pair_mean = pairs[pair][1]
                pair_mean_new = (x_0 + (pairs[pair][0] - 1) * pair_mean) / pairs[pair][0]
                pairs[pair][1] = pair_mean_new
                #pair_var = ((pairs[pair][0] - 2) * pairs[pair][2] + (pairs[pair][0] - 1) *
                #            ((pair_mean - pair_mean_new) ** 2) + ((x_0 - pair_mean_new) ** 2)) / (pairs[pair][0] - 1)
                #pair_st_dev = pair_var ** 0.5
                #pairs[pair][2] = pair_var
        pred['TOK_MPTAP'] = pairs[pair][1]
        #print(pairs[pair][1])
        #print(mean)
        #print(st_dev)

        if pairs[pair][1] > mean + st_dev * 0.12:
            c += 1
            #print(str(True) + str(c))
            pred['TOK_MPTAP_is_cut'] = True
        mptap_mean_new = (pairs[pair][1] + i * mptap_mean) / (i + 1)
        mptap_var = ((i - 1) * var + i * ((mptap_mean - mptap_mean_new) ** 2) + ((pairs[pair][1] - mptap_mean_new) ** 2)) / i
        mptap_st_dev = mptap_var ** 0.5
        mptap_mean = mptap_mean_new

        if x_1 > mean + st_dev * 0.8:
            current['TOK_TAP_is_cut'] = True
        mean_new = (x_1 + i * mean) / (i+1)
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
df.loc[df['time_diff'].nlargest(k).index, 'TAP_is_cut'] = True
df['TAP_discovered_case'] = df['TAP_is_cut'].shift(1).cumsum().fillna(0)
df['next_guess_col'] = df['case'].shift(-1)
df['new_guess_col'] = (df['next_guess_col']!=df['case'])|(df['next_guess_col'].isna())

'''
tap = 0
tok = 0
both = 0
for i in range(len(df)):
    if df.iloc[i]['TAP_is_cut']:
        if df.iloc[i]['TOK_TAP_is_cut']:
            both += 1
        else:
            tap += 1
    else:
        if df.iloc[i]['TOK_TAP_is_cut']:
            tok += 1

print("tap: " + str(tap))
print("tok: " + str(tok))
print("both: " + str(both))


tap = 0
tap_wrong = 0
tok = 0
tok_wrong = 0
traces = 0
for i in range(len(df)):
    if df.iloc[i]['last_trace']:
        traces += 1
        if df.iloc[i]['TAP_is_cut']:
            tap += 1
        if df.iloc[i]['TOK_TAP_is_cut']:
            tok += 1
    else:
        if df.iloc[i]['TAP_is_cut']:
            tap_wrong += 1
        if df.iloc[i]['TOK_TAP_is_cut']:
            tok_wrong += 1

print("tap_corr: " + str(tap))
print("tok_corr: " + str(tok))
print("traces: " + str(traces))
print("tap_wro: " + str(tap_wrong))
print("tok_wro: " + str(tok_wrong))
'''

# METHOD 2: LCPAP (using the mean time between pairs of events)
# Same as method 1, but we replace the true time difference
# by the average time difference per pair of events
df['next_activity'] = df[activity].shift(-1)
df['pair'] = df[activity].astype(str) + '_' + df['next_activity'].astype(str)
mapping = df.groupby('pair')['time_diff'].mean()
df['MPTAP'] = df['pair'].map(mapping)
df['MPTAP_is_cut'] = False
df.loc[df['MPTAP'].nlargest(k).index, 'MPTAP_is_cut'] = True
df['MPTAP_discovered_case'] = df['MPTAP_is_cut'].shift(1).cumsum().fillna(0)


mptap = 0
tok = 0
both = 0
for i in range(len(df)):
    if df.iloc[i]['MPTAP_is_cut']:
        if df.iloc[i]['TOK_MPTAP_is_cut']:
            both += 1
        else:
            mptap += 1
    else:
        if df.iloc[i]['TOK_MPTAP_is_cut']:
            tok += 1

print("mptap: " + str(mptap))
print("tok: " + str(tok))
print("both: " + str(both))


mptap = 0
mptap_wrong = 0
tok = 0
tok_wrong = 0
traces = 0
for i in range(len(df)):
    if df.iloc[i]['last_trace']:
        traces += 1
        if df.iloc[i]['MPTAP_is_cut']:
            mptap += 1
        if df.iloc[i]['TOK_MPTAP_is_cut']:
            tok += 1
    else:
        if df.iloc[i]['MPTAP_is_cut']:
            mptap_wrong += 1
        if df.iloc[i]['TOK_MPTAP_is_cut']:
            tok_wrong += 1

print("mptap_corr: " + str(mptap))
print("tok_corr: " + str(tok))
print("traces: " + str(traces))
print("mptap_wro: " + str(mptap_wrong))
print("tok_wro: " + str(tok_wrong))

print(df.head(1000).to_string())
#print(pairs)
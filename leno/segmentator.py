import pandas as pd
import warnings
import pm4py
import time

import strong_connected_comp
#import directly_follows_graph

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
# if back to back copy events from Chrome and OS-Clipboard, delete OS-Clipboard copy
def deleteChromeClipboardCopy(row1, row2):
    if row1['targetApp'] == "Chrome" and row1['eventType'] == "copy":
        if row2['targetApp'] == "OS-Clipboard" and row2['eventType'] == "copy":
            return True
    return False

def mergeGetCellCopy(row1, row2, row3):
    if row1['eventType'] == "getCell":
        if row2['eventType'] == "editField" or row2['eventType'] == "getRange" or row2['eventType'] == "getCell":
            if row3['targetApp'] == "OS-Clipboard" and row3['eventType'] == "copy":
                return True
    return False

def mergeEditCellCopy(row1, row2, row3):
    if row1['eventType'] == "editField":
        if row2['eventType'] == "editField" or row2['eventType'] == "getRange" or row2['eventType'] == "getCell":
            if row3['targetApp'] == "OS-Clipboard" and row3['eventType'] == "copy":
                return True
    return False

def mergeGetRangeCopy(row1, row2, row3):
    if row1['eventType'] == "getRange":
        if row2['eventType'] == "editField" or row2['eventType'] == "getRange" or row2['eventType'] == "getCell":
            if row3['targetApp'] == "OS-Clipboard" and row3['eventType'] == "copy":
                return True
    return False

def preprocessV1(log):
    df = pd.DataFrame()
    for row in log:
        df.append(row)
        while len(df) > 1 and deleteChromeClipboardCopy(df.loc[-2], df.loc[-1]):
            # delete OS-Clipboard line (last row)
            df = df.iloc[:-1]
        while len(df) > 2 and (mergeGetCellCopy(df.loc[-3], df.loc[-2], df.loc[-1]) or mergeEditCellCopy(df.loc[-3], df.loc[-2], df.loc[-1])):
            df.iloc[-1, 1:2] = df.iloc[-2, 1:2]
            df.iloc[-1, 3] = "copyCell"
            df.iloc[-1, 6:] = df.iloc[-2, 6:]
        while len(df) > 2 and mergeGetRangeCopy(df.loc[-3], df.loc[-2], df.loc[-1]):
            df.iloc[-1, 1:2] = df.iloc[-2, 1:2]
            df.iloc[-1, 3] = "copyRange"
            df.iloc[-1, 6:] = df.iloc[-2, 6:]
        while len(df) > 0 and (df.iloc[-1]['eventType'] == "getRange" or df.iloc[-1]['eventType'] == "getCell"):
            df = df.iloc[:-1]

    return df
    '''

def preprocess(log):
    df = pd.DataFrame()
    activities = []
    edges = {}
    header = {}

    def mergeNavigationCellCopy(row1, row2):

        # if back to back copy events from Chrome and OS-Clipboard, delete OS-Clipboard copy
        # if row1['targetApp'] == "Chrome" and row1['eventType'] == "copy":
        #    row2 = row2[0:0]

        # merge getCell or editField and copy events
        if row1['eventType'] == "getCell" or row1['eventType'] == "editCell":
            row1['timeStamp'] = row2['timeStamp']
            row1['content'] = row2['content']
            row1['eventType'] = "copyCell"
            # row2 = row2[0:0]

        # merge getRange and copy events
        elif row1['eventType'] == "getRange":
            row1['timeStamp'] = row2['timeStamp']
            row1['content'] = row2['content']
            row1['eventType'] = "copyRange"
            # row2 = row2[0:0]

        return row1

    def segment(row, row_pre):
        dfg.update(row, row_pre)

        return row

    for i in range(len(log) - 1):
        #print(i)
        row1 = log.iloc[i]
        row2 = log.iloc[i+1]

        if row1['eventType'] == "copy" or row1['eventType'] == "clickTextField":
            continue
        elif row2['eventType'] == "copy":
            row = mergeNavigationCellCopy(row1, row2)
        else:
            row = row1

        df = df.append(row)

        if row['eventType'] not in activities:
            activities.append(row['eventType'])

        if i in {3, 30, 40, 100, 550, 1000, 2000}:
            dfg, start_activities, end_activities = pm4py.discover_dfg_typed(df, case_id_key='userID',
                                                                         activity_key='eventType',
                                                                         timestamp_key='timeStamp')
            pm4py.save_vis_performance_dfg(dfg, start_activities, end_activities, 'perf_dfg_' + str(i) + '.svg')

        #if len(df) == 1:
        #    dfg.update()
        #elif len(df) > 1:
        #    row = segment(df.iloc[-1], df.iloc[-2])

    dfg, start_activities, end_activities = pm4py.discover_dfg_typed(df, case_id_key='userID', activity_key='eventType',
                                                               timestamp_key='timeStamp')

    print("dfg:")
    print(dfg)
    print("activities:")
    print(activities)
    for edge in dfg:
        src = edge[0]
        tgt = edge[1]
        if src in edges:
            edges[src].append(tgt)
        else:
            edges[src] = [tgt]
    print("edges:")
    print(edges)
    sccs = strong_connected_comp.discover_scc(activities, edges)
    for s in sccs:
        print(s[0])
        print(s[1])
        header[s[1]] = s[0]
        print(header)
        # print(s)
        # print(nx.immediate_dominators(nx.DiGraph(edges), activities[0]).items())

    print(start_activities)
    print(end_activities)
    pm4py.save_vis_performance_dfg(dfg, start_activities, end_activities, 'perf_dfg.svg')

    case = 0
    df = df.reset_index()
    for i in range(1, len(df)):
        for h in header:
            if df.at[i, 'eventType'] == h and df.at[i - 1, 'eventType'] in header[h]:
                case += 1
                break
        df.at[i, 'index'] = str(case)

    return df

log_name = 'StudentRecord'
with open(log_name + '.csv') as log:
    print("Reading CSV...")
    t = time.time()
    df = pd.read_csv(log, sep=',')
    print("Reading CSV completed: " + str((time.time() - t) * 1000) + " ms")
    df = df.sort_values(by=['timeStamp'], ignore_index=True)
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    t = time.time()
    print("Preprocessing Log...")
    df = preprocess(df)
    print("Preprocessing Log completed: " + str((time.time() - t) * 1000) + " ms")
    df.to_csv(log_name + '_preprocessed.csv', sep=',', index=False)

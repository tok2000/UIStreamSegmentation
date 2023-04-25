import math

import pandas as pd
import warnings
import pm4py
import time

import strong_connected_comp

# import directly_follows_graph

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

        # merge getCell or editField and copy events
        if row1['eventType'] == "getCell" or row1['eventType'] == "editCell":
            row1['timeStamp'] = row2['timeStamp']
            row1['content'] = row2['content']
            row1['eventType'] = "copyCell"

        # merge getRange and copy events
        elif row1['eventType'] == "getRange":
            row1['timeStamp'] = row2['timeStamp']
            row1['content'] = row2['content']
            row1['eventType'] = "copyRange"

        return row1

    def mergeCopyPaste(row1, row2):
        row = row2
        row['eventType'] = 'copy_paste'
        for att in row.keys():
            if pd.isnull(row[att]):
                row[att] = row1[att]
        return row

    def segment(row, row_pre):
        dfg.update(row, row_pre)

        return row

    cid = 1
    for i in range(len(log) - 3):
        row1 = log.loc[i]
        row2 = log.loc[i + 1]

        if i in {25, 50, 100, 500, 1000, 2000}:
            dfg, start_activities, end_activities = pm4py.discover_dfg_typed(df, case_id_key='userID',
                                                                             activity_key='eventType_concrete',
                                                                             timestamp_key='timeStamp')
            pm4py.save_vis_performance_dfg(dfg, start_activities, end_activities, 'perf_dfg_' + str(i) + '.svg')

        event1 = row1['eventType']
        if event1 == 'copy' or event1 == 'clickTextField' or event1 == 'form_submit' or event1 == 'ignore':
            continue
        elif row2['eventType'] == 'copy':
            row = mergeNavigationCellCopy(row1, row2)
            #row2 = log.loc[i + 2]
            #if row2['eventType'] == 'paste':
            #    row = mergeCopyPaste(row, row2)
            #    log.loc[i + 2, 'eventType'] = 'ignore'
            #elif row2['eventType'] == 'clickTextField' or row2['eventType'] == 'editField':
            #    row3 = log.loc[i + 3]
            #    if row3['eventType'] == 'paste':
            #        row = mergeCopyPaste(row, row3)
            #        log.loc[i + 3, 'eventType'] = 'ignore'
        else:
            row = row1

        # if pd.notnull(row['target.name']) and row['eventType'] != "paste":
        #    row['eventType_concrete'] = str(row['eventType']) + '[' + str(row['target.name']) + ']'
        if row['eventType'] == "clickButton" and pd.notnull(row['target.type']):
            row['eventType_concrete'] = str(row['eventType']) + '[' + str(row['target.type']) + ']'
        else:
            row['eventType_concrete'] = row['eventType']

        if row['target.innerText'] in ['Add another response', 'Add another response.']:
            cid += 1

        row = pd.concat([pd.Series(data={'case_id': cid}), row])

        # print(row.to_string())
        # print()

        df = df.append(row, ignore_index=True)

        if row['eventType_concrete'] not in activities:
            activities.append(row['eventType_concrete'])

        # if len(df) == 1:
        #    dfg.update()
        # elif len(df) > 1:
        #    row = segment(df.iloc[-1], df.iloc[-2])

    df = df.append(log[-2:-1], ignore_index=True)
    df.loc[len(df) - 1, 'case_id'] = cid

    print(df.head(100).to_string())
    dfg, start_activities, end_activities = pm4py.discover_dfg_typed(df, case_id_key='userID',
                                                                     activity_key='eventType_concrete',
                                                                     timestamp_key='timeStamp')

    events3 = {}
    for k in range(len(df) - 5):
        changed = False
        r1 = df.loc[k, 'eventType_concrete']
        r2 = df.loc[k + 1, 'eventType_concrete']
        r3 = df.loc[k + 2, 'eventType_concrete']
        r4 = df.loc[k + 3, 'eventType_concrete']
        r5 = df.loc[k + 4, 'eventType_concrete']
        triplet = (r1, r2, r3, r4, r5)
        for trip in events3:
            if set(trip).issubset(set(triplet)):
                events3[trip] += 1
                changed = True
                break
        if not changed:
            events3[triplet] = 1

    print("triplets:")
    print(events3)

    df.drop(['eventType_concrete'], axis=1, inplace=True)
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


log_name = 'Reimbursement'
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

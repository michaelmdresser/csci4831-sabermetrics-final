import pandas as pd
from pybaseball.playerid_lookup import get_lookup_table
from pybaseball import playerid_reverse_lookup

###
### USAGE
###
### from util import make_pitcher_df, make_efp_series, make_batter_df, make_efb_series
### from pybaseball import batting_stats, statcast
###
### statcast_data = statcast(???)
### fangraphs_batting_data = batting_data(???)
###
### efp = make_efp_series(make_pitcher_df(statcast))
### efb = make_efb_series(make_batter_df(fangraphs_batting_data, statcast_data))
###

# gets the number of batters a pitcher faced where the pitch count went above 5
# requires a statcast df
def get_pca5_series_pitcher(df):
    pab = df.groupby(["pitcher", "game_date", "batter", "inning"]).size()
    return pab.loc[pab > 5].to_frame().reset_index().groupby("pitcher").size()

# gets the plate appearances a batter had where the pitch count went above 5
# requires a statcast df
def get_pca5_series_batter(df):
    bab = df.groupby(["batter", "game_date", "inning"]).size()
    return bab.loc[bab > 5].to_frame().reset_index().groupby("batter").size()

# gets the number of batters a pitcher faced
# requires a statcast df
def get_bf(df):
    return df.loc[df["events"].notna()].groupby(["pitcher"]).size()

# returns tuple (release_speed_mean, effective_speed_mean, pfx_x_mean, pfx_z_mean) for each pitcher
# requires a statcast df
def get_movement_series(df):
    d = df.groupby("pitcher").agg({"release_speed": "mean",
                                 "effective_speed": "mean",
                                 "pfx_x": "mean", 
                                 "pfx_z": "mean"}).abs().rename(index=str, columns={"release_speed": "release_speed_mean",
                                                                 "effective_speed": "effective_speed_mean",
                                                                 "pfx_x": "pfx_x_mean",
                                                                 "pfx_z": "pfx_z_mean"})
    return (d["release_speed_mean"], d["effective_speed_mean"], d["pfx_x_mean"], d["pfx_z_mean"])

# gets the number of a specific type of event that occurred to each pitcher
# requires a statcast df
def get_event_series_pitcher(df, event):
    return df.loc[(df["events"] == event)].groupby("pitcher").size()

# gets the number of a specific type of event that occurred to each batter
# requires a statcast df
def get_event_series_batter(df, event):
    return df.loc[(df["events"] == event)].groupby("batter").size()

# gets the number of IP for every pitcher
# this only estimates ip, it is not perfect
# requires a statcast df
def get_ip_series(df):
    return df.groupby(["pitcher", "game_date"])["inning"].nunique().to_frame().reset_index().groupby("pitcher")["inning"].sum()

# gets the WHIP for every pitcher
# requires a statcast df
def get_whip_series(df):
    walks = df.loc[(df["events"] == "walk")].groupby("pitcher").size()
    hits = df.loc[(df["events"] == "single") |
                        (df["events"] == "double") |
                        (df["events"] == "triple") |
                        (df["events"] == "home_run")].groupby("pitcher").size()
    stuff = pd.concat([get_ip_series(df), walks, hits], keys=["ip", "walks", "hits"], axis=1)
    return (stuff["walks"] + stuff["hits"]) / stuff["ip"]

# removes pitchers with low batters faced
# requires a statcast df
def remove_infreq_pitchers(df, minimum_bf=70):
    bf = get_bf(df)

    return df.loc[(df["pitcher"].isin(bf.loc[bf >= minimum_bf].index))]

# builds a dataframe of all pitchers with the pitcher as the index that contains
# all of the statistics we care about to calculate EFP
# requires a statcast df
def make_pitcher_df(df):
    df = remove_infreq_pitchers(df)
    _, avg_speed, avg_x, avg_z = get_movement_series(df)
    return pd.concat([get_event_series_pitcher(df, "strikeout"),
                      get_event_series_pitcher(df, "home_run"),
                      get_event_series_pitcher(df, "walk"), # I don't think we can do IBB
                      get_pca5_series_pitcher(df),
                      get_bf(df),
                      get_whip_series(df),
                      avg_speed,
                      avg_x,
                      avg_z
                     ],
                     keys=["SO", "HR", "BB", "PCA5", "BF", "WHIP", "avg_speed", "avg_x", "avg_z"],
                     axis=1)

# Takes a pitcher_df from make_pitcher_df and returns a series (index = pitcher) of
# each pitcher's EFP stat
def make_efp_series(pitcher_df):
    df = pitcher_df
    # WE DON'T HAVE IBB, only BB
    return (1.0*df["SO"] - (0.5*df["PCA5"] + 3.0*df["HR"] + 3.0*df["BB"])) / df["BF"] - \
            1.0*df["WHIP"] + 0.1*df["avg_z"] + 0.1*df["avg_x"] + 0.3*df["avg_speed"]


# stolen from github.com/jldbc/pybaseball and heavily modified for this application
# almost the same functionality but faster and has hardcoded exceptions for
# some weird player names from 2018
def playerid_lookup_c(last, first=None, year=None, lookup_table=None):
    if lookup_table is None:
        raise Exception("this version of playerid lookup requires a pre-built lookup table")

    # force input strings to lowercase
    last = last.lower()
    if first:
        first = first.lower()
    table = lookup_table
    
    if first is None:
        if year is not None:
            results = table.loc[(table['name_last'] == last) &
                                (table['mlb_played_first'] <= year) &
                                (table['mlb_played_last'] >= year)]
        else:
            results = table.loc[table['name_last'] == last]
    else:
        if year is not None:
            results = table.loc[(table['name_last'] == last) & (table['name_first'] == first) &
                                (table['mlb_played_first'] <= year) &
                                (table['mlb_played_last'] >= year)]
        else:
            results = results = table.loc[(table['name_last'] == last) & (table['name_first'] == first)]

        iterations = 0
        while len(results) == 0 and iterations < 10:
            iterations += 1
            if 'jr.' in last:
                last = last.replace('jr.', '').strip()
            elif 'jr' in last:
                last = last.replace('jr', '').strip()
            elif '.' in first and ' ' in first:
                first = first.replace(' ', '').replace('.', '')
            elif '.' in first:
                first = first[:first.index('.') + 1] + ' ' + first[first.index('.') + 1:]
            elif first == 'nicholas':
                first = 'nick'
            elif first == 'yolmer' and last == 'sanchez':
                first = 'carlos'
            elif first == 'raffy' and last == 'lopez':
                first = 'rafael'
            elif last == 'ervin':
                first = 'phil'
            elif last == 'wheeler' and first == 'zack':
                first = 'zach'
            elif last == 'joyce' and first == 'matt':
                first = 'matthew'
            elif last == 'vogelbach' and first == 'daniel':
                first = 'dan'
            elif last == 'kang':
                first = 'jung ho'
            elif last == 'urshela':
                first = 'gio'
            elif last == 'poncedeleon':
                last = 'ponce de leon'
            elif last == 'bowman' and first == 'matt':
                first = 'matthew'
            elif last == 'chargois':
                first = 'j. t.'
            elif last == 'boyd' and first == 'matthew':
                first = 'matt'
            elif last == 'gosselin' and first == 'phil':
                first = 'philip'
            elif last == 'guerra' and first == 'javier':
                first = 'javy'
            elif last == 'delmonico' and first == 'nicky':
                first = 'nick'
            elif last == 'wilkerson' and first == 'steve':
                first = 'stevie'
            elif first == 'john' and last == 'ryan murphy':
                first = 'j. r.'
                last = 'murphy'
            else:
                for name_bit in last.split(' '):
                    if len(name_bit) <= 3:
                        last = last.replace(name_bit, '').strip()

            if year is not None:
                results = table.loc[(table['name_last'] == last) & (table['name_first'] == first) &
                                    (table['mlb_played_first'] <= year) &
                                    (table['mlb_played_last'] >= year)]
            else:
                results = results = table.loc[(table['name_last'] == last) &
                                              (table['name_first'] == first)]
    
    
    #results[['key_mlbam', 'key_fangraphs', 'mlb_played_first', 'mlb_played_last']] = results[['key_mlbam', 'key_fangraphs', 'mlb_played_first', 'mlb_played_last']].astype(int) # originally returned as floats which is wrong
    results = results.reset_index().drop('index', 1)
    return results

# takes a player's name and returns their mlbam key/id
def get_mlbam_from_name(last, first=None, year=None, lookup_table=None):
    if lookup_table is None:
        raise Exception("get_mlbam_from_name needs a pre-built lookup table to avoid speed problems")

    try:
        return playerid_lookup_c(last, first, year=year, lookup_table=lookup_table).dropna().reset_index(drop=True)["key_mlbam"].iloc[0]
    except IndexError as e:
        if '.' in first and ' ' in first:
            first = first.replace(' ', '').replace('.', '')
        elif '.' in first:
            first = first[:first.index('.') + 1] + ' ' + first[first.index('.') + 1:]
        elif first.lower() == 'nicholas':
            first = 'nick'
        elif first.lower() == 'yolmer' and last.lower() == 'sanchez':
            first = 'carlos'
        else:
            #raise Exception("your name bad: %s, %s" % (last, first))
            print("bad name: %s, %s" % (last, first))
            return -1
        
        return get_mlbam_from_name(last, first)
        return playerid_lookup_c(last, first).dropna().reset_index(drop=True)["key_mlbam"].iloc[0]
        # j.d. martinez -> j. d. martinez

# Takes a Fangraphs batting_stats df (from pybaseball import batting_stats) and adds every player's
# mlbam key to their row. Doesn't work for all names, do not consider this foolproof, only
# good enough.
# Side note (related to unreliability): If fangraphs has a unique identifier, WHY doesn't batting_stats
# include that identifier with the data????
def add_mlbam_to_fg(df_fg):
    lookup_table = get_lookup_table()
    def get_last_first(name):
        s = name.split(' ')
        last, first = ' '.join(s[1:]), s[0]
        return last, first
    
    def get_mlbam(name):
        last, first = get_last_first(name)
        return get_mlbam_from_name(last, first, lookup_table=lookup_table)

    
    df_fg["key_mlbam"] = df_fg["Name"].apply(get_mlbam)
    return df_fg

# Removes batters from a fangraphs DF whose AB is lower than a value
def remove_infreq_batters(df_fangraphs, minimum_ab=50):
    return df_fangraphs.loc[(df_fangraphs["AB"] >= minimum_ab)]

# takes a fangraphs df from pybaseball.batting_stats and a statcast df
# and adds mlbam keys and a pca5 column to the fangraphs df
# also indexes the df by mlbam key
def make_batter_df(df_fangraphs, df_statcast):
    df_fg = remove_infreq_batters(df_fangraphs).copy()
    df_fg = add_mlbam_to_fg(df_fg)
    pca5 = get_pca5_series_batter(df_statcast).rename("PCA5")
    df_fg = df_fg.merge(pca5.to_frame(), left_on="key_mlbam", right_on="batter")
    df_fg = df_fg.set_index("key_mlbam")
    return df_fg

# takes a batter df from make_batter_df and returns a series indexed by
# batter mlbam key of batter EFB
def make_efb_series(batter_df):
    df = batter_df
    return (((1.0 * df["HR"] + 0.8*df["3B"] + 0.5*df["2B"] + 0.2*df["1B"] + 0.5*df["SB"] - \
           (0.4*df["SO"] + 0.2*df["BB"] + 0.2*df["PCA5"])) / df["PA"]) + 2) * 10

# makes efb json as taken by the efb variable in efb.html
def make_efb_json(df_fangraphs, df_statcast):
    b_df = make_batter_df(b, statcast)
    efb_s = make_efb_series(b_df).sort_values(ascending=False)
    return efb_s.rename("EFB").to_frame().merge(b_df[["Name", "Team", "HR", "3B", "2B", "1B", "SB", "SO", "BB", "PCA5", "PA"]], left_index=True, right_index=True).to_json(orient="columns")

# makes efp json as taken by the efp variable in efp.html
def make_efp_json(statcast):
    efp_s = make_efp_series(make_pitcher_df(statcast))
    ids = []
        for i in efp_s.index:
            ids.append(i)

    pitcher_ids = playerid_reverse_lookup(ids)

    p_df = make_pitcher_df(statcast)
    efp_s = make_efp_series(p_df).rename("EFP")
    combined = p_df.merge(efp_s.to_frame(), left_index=True, right_index=True)
    with_ids = combined.merge(pitcher_ids[["name_last", "name_first", "key_mlbam"]], left_index=True, right_on="key_mlbam").set_index("key_mlbam", drop=True)

    return with_ids.to_json(orient="columns")

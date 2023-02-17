
# python -m streamlit run C:\Users\user1\Downloads\AIFmaps_converter\AIF_convert.py

# imports
import streamlit as st
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-talk")

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import glob
import json

def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")

import os.path
import requests
from datetime import datetime
from pathlib import Path
import re
#import networkx as nx
import plotly.express as px

# https://github.com/roryduthie/AIF_Converter
def get_graph_url(node_path):
    try:
        jsn_string = requests.get(node_path).text
        strng_ind = jsn_string.index('{')
        n_string = jsn_string[strng_ind:]
        dta = json.loads(n_string)
    except:
        st.error(f'File was not found: {node_path}')
    return dta



@st.cache(allow_output_mutation=True)
def RetrieveNodes(node_list, type_aif='old'):
  df_all_loc = pd.DataFrame(columns = ['locution', 'connection', 'illocution', 'id_locution', 'id_connection', 'id_illocution', 'nodeset_id'])
  df_all = pd.DataFrame(columns = ['premise', 'connection', 'conclusion', 'id_premise', 'id_connection', 'id_conclusion', 'nodeset_id'])

  for map in node_list[:]:
    try:
      with open(map, 'r') as f:
        map1 = json.load(f)

      if type_aif == 'new':
        df_nodes = pd.DataFrame(map1['AIF']['nodes'])
        df_edge = pd.DataFrame(map1['AIF']['edges'])
      else:
        df_nodes = pd.DataFrame(map1['nodes'])
        df_edge = pd.DataFrame(map1['edges'])

      tto1l = []
      tfrom1l = []
      tto2l = []
      tfrom2l = []
      connect_idsl = []
      loc_idsl = []
      illoc_idsl = []
      nodeset_idsl = []

      match_nodeset = re.split('nodeset', str(map))
      match_nodeset = match_nodeset[-1][:4]

      tto1i = []
      tfrom1i = []
      tto2i = []
      tfrom2i = []
      connect_idsi = []
      loc_idsi = []
      illoc_idsi = []
      nodeset_idsi = []
      rels = ['MA', 'CA', 'RA', 'PA']

      for id1 in df_edge.index:
        for id2 in df_edge.index:

          id_from1 =  df_edge.loc[id1, 'fromID']
          id_to1 =  df_edge.loc[id1, 'toID']

          id_from2 =  df_edge.loc[id2, 'fromID']
          id_to2 =  df_edge.loc[id2, 'toID']

          if id_to1 == id_from2:
            # locutions
            if (df_nodes[ (df_nodes.nodeID == id_from2) ]['type'].iloc[0] == 'YA') and (df_nodes[ (df_nodes.nodeID == id_to2) ]['type'].iloc[0] == 'I')  and (df_nodes[ (df_nodes.nodeID == id_from1) ]['type'].iloc[0] == 'L'):
              d1 = df_nodes[ (df_nodes.nodeID == id_from1) ]
              d2 = df_nodes[ (df_nodes.nodeID == id_to1) ]

              d11 = df_nodes[ (df_nodes.nodeID == id_from2) ]
              d22 = df_nodes[ (df_nodes.nodeID == id_to2) ]

              tto1l.append(d2['text'].iloc[0])
              tfrom1l.append(d1['text'].iloc[0])

              tto2l.append(d22['text'].iloc[0])
              tfrom2l.append(d11['text'].iloc[0])

              connect_idsl.append(id_to1)
              loc_idsl.append(d1['nodeID'].iloc[0])
              illoc_idsl.append(d22['nodeID'].iloc[0])

              nodeset_idsl.append(match_nodeset)

            # args
            if (df_nodes[ (df_nodes.nodeID == id_to1) ]['type'].iloc[0] in rels) and (df_nodes[ (df_nodes.nodeID == id_from1) ]['type'].iloc[0] != 'YA') and (df_nodes[ (df_nodes.nodeID == id_to2) ]['type'].iloc[0] == 'I'):
              d1 = df_nodes[ (df_nodes.nodeID == id_from1) ]
              d2 = df_nodes[ (df_nodes.nodeID == id_to1) ]

              d11 = df_nodes[ (df_nodes.nodeID == id_from2) ]
              d22 = df_nodes[ (df_nodes.nodeID == id_to2) ]

              tto1i.append(d2['text'].iloc[0])
              tfrom1i.append(d1['text'].iloc[0])

              tto2i.append(d22['text'].iloc[0])
              tfrom2i.append(d11['text'].iloc[0])

              connect_idsi.append(id_to1)
              loc_idsi.append(d1['nodeID'].iloc[0])
              illoc_idsi.append(d22['nodeID'].iloc[0])
              nodeset_idsi.append(match_nodeset)

      df1 = pd.DataFrame({
          'locution': tfrom1l,
          'connection': tto1l,
          'illocution': tto2l,
          'id_locution': loc_idsl,
          'id_connection': connect_idsl,
          'id_illocution': illoc_idsl,
          'nodeset_id': nodeset_idsl,
      })

      df_all_loc = pd.concat( [df_all_loc, df1], axis=0, ignore_index=True )

      df2 = pd.DataFrame({
          'premise': tfrom1i,
          'connection': tto1i,
          'conclusion': tto2i,
          'id_premise': loc_idsi,
          'id_connection': connect_idsi,
          'id_conclusion': illoc_idsi,
          'nodeset_id': nodeset_idsi,
      })
      df_all = pd.concat( [df_all, df2], axis=0, ignore_index=True )

    except:
      #print(f'Except {map} ')
      continue
  return df_all_loc, df_all



@st.cache(allow_output_mutation=True)
def RetrieveNodesOnline(map1, nodeset_id_str, type_aif='old'):
  try:
      if type_aif == 'new':
        df_nodes = pd.DataFrame(map1['AIF']['nodes'])
        df_edge = pd.DataFrame(map1['AIF']['edges'])
      else:
        df_nodes = pd.DataFrame(map1['nodes'])
        df_edge = pd.DataFrame(map1['edges'])

      tto1l = []
      tfrom1l = []
      tto2l = []
      tfrom2l = []
      connect_idsl = []
      loc_idsl = []
      illoc_idsl = []
      nodeset_idsl = []

      match_nodeset = nodeset_id_str

      tto1i = []
      tfrom1i = []
      tto2i = []
      tfrom2i = []
      connect_idsi = []
      loc_idsi = []
      illoc_idsi = []
      nodeset_idsi = []
      rels = ['MA', 'CA', 'RA', 'PA']

      for id1 in df_edge.index:
        for id2 in df_edge.index:

          id_from1 =  df_edge.loc[id1, 'fromID']
          id_to1 =  df_edge.loc[id1, 'toID']

          id_from2 =  df_edge.loc[id2, 'fromID']
          id_to2 =  df_edge.loc[id2, 'toID']

          if id_to1 == id_from2:
            # locutions
            if (df_nodes[ (df_nodes.nodeID == id_from2) ]['type'].iloc[0] == 'YA') and (df_nodes[ (df_nodes.nodeID == id_to2) ]['type'].iloc[0] == 'I')  and (df_nodes[ (df_nodes.nodeID == id_from1) ]['type'].iloc[0] == 'L'):
              d1 = df_nodes[ (df_nodes.nodeID == id_from1) ]
              d2 = df_nodes[ (df_nodes.nodeID == id_to1) ]

              d11 = df_nodes[ (df_nodes.nodeID == id_from2) ]
              d22 = df_nodes[ (df_nodes.nodeID == id_to2) ]

              tto1l.append(d2['text'].iloc[0])
              tfrom1l.append(d1['text'].iloc[0])

              tto2l.append(d22['text'].iloc[0])
              tfrom2l.append(d11['text'].iloc[0])

              connect_idsl.append(id_to1)
              loc_idsl.append(d1['nodeID'].iloc[0])
              illoc_idsl.append(d22['nodeID'].iloc[0])

              nodeset_idsl.append(match_nodeset)

            # args
            if (df_nodes[ (df_nodes.nodeID == id_to1) ]['type'].iloc[0] in rels) and (df_nodes[ (df_nodes.nodeID == id_from1) ]['type'].iloc[0] != 'YA') and (df_nodes[ (df_nodes.nodeID == id_to2) ]['type'].iloc[0] == 'I'):
              d1 = df_nodes[ (df_nodes.nodeID == id_from1) ]
              d2 = df_nodes[ (df_nodes.nodeID == id_to1) ]

              d11 = df_nodes[ (df_nodes.nodeID == id_from2) ]
              d22 = df_nodes[ (df_nodes.nodeID == id_to2) ]

              tto1i.append(d2['text'].iloc[0])
              tfrom1i.append(d1['text'].iloc[0])

              tto2i.append(d22['text'].iloc[0])
              tfrom2i.append(d11['text'].iloc[0])

              connect_idsi.append(id_to1)
              loc_idsi.append(d1['nodeID'].iloc[0])
              illoc_idsi.append(d22['nodeID'].iloc[0])
              nodeset_idsi.append(match_nodeset)

      df_all_loc = pd.DataFrame({
          'locution': tfrom1l,
          'connection': tto1l,
          'illocution': tto2l,
          'id_locution': loc_idsl,
          'id_connection': connect_idsl,
          'id_illocution': illoc_idsl,
          'nodeset_id': nodeset_idsl})

      df_all = pd.DataFrame({
          'premise': tfrom1i,
          'connection': tto1i,
          'conclusion': tto2i,
          'id_premise': loc_idsi,
          'id_connection': connect_idsi,
          'id_conclusion': illoc_idsi,
          'nodeset_id': nodeset_idsi})

  except:
    st.error('Error loading nodeset')
  return df_all_loc, df_all




##################### page config  #####################
st.set_page_config(page_title="AIF Converter", layout="wide") # centered wide

#####################  page content  #####################
st.title("AIF Converter")
add_spacelines(3)


maps = glob.glob(r"/maps/*.json")

directory = "tem_maps"
parent_dir = "/"
temp_path = os.path.join(parent_dir, directory)
############################################################################



#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Parameters of analysis")
    add_spacelines(1)
    type_aif = st.radio("Choose AIF version", ('Old', 'New'))

    add_spacelines(2)
    own_files_bool = st.radio('Choose corpora', ('Sample corpus', 'Insert files'))
    if own_files_bool == 'Sample corpus':
        df_all_loc, df_all = RetrieveNodes(maps[:], type_aif = str(type_aif).lower())

    else:
        own_files = st.radio('Chose method of uploading files', ('Upload files', 'Nodeset ID from AIF'))

        if own_files == 'Upload files':
            uploaded_json = st.file_uploader('Choose files', type = 'json', accept_multiple_files = True)
            if len(uploaded_json) < 1:
                st.stop()
            elif len(uploaded_json) > 1:
                maps = glob.glob(r"/maps_up/*.json")
                if len(maps) > 0:
                    for f in maps:
                        os.remove(f)
                for file in uploaded_json:
                  with open(os.path.join(r"/maps_up", file.name), "wb") as f:
                      f.write(file.getbuffer())
                      st.write(f'{file.name} saved sucessfully')
            maps = glob.glob(r"/maps_up/*.json")
            df_all_loc, df_all = RetrieveNodes(maps[:], type_aif = str(type_aif).lower())

        elif own_files == 'Nodeset ID from AIF':
            nodeset_id_input = st.text_input("Insert nodeset ID from AIFb", "10453")
            if len(nodeset_id_input) < 1:
                st.stop()
            elif len(nodeset_id_input) > 1:
                file_json_nodeset = get_graph_url(f'http://www.aifdb.org/json/{nodeset_id_input}')
                #if not os.path.exists(temp_path):
                    #os.mkdir(temp_path)
                #with open(os.path.join(temp_path, 'nodeset'+str(nodeset_id_input)+'.json'), "w") as f:
                    #json.dump(file_json_nodeset, f)
                    #st.write(f'{nodeset_id_input} saved sucessfully')
                #maps = glob.glob("/tem_maps/*.json")
                df_all_loc, df_all = RetrieveNodesOnline(file_json_nodeset, nodeset_id_str = nodeset_id_input, type_aif = str(type_aif).lower())


#  *********************** sidebar  *********************



#  *********************** PAGE content  *********************

#st.dataframe(df_all)
ids_linked = df_all[df_all.id_connection.duplicated()].id_connection.unique()
if len(ids_linked) > 0:
    df_all.loc[df_all.id_connection.isin(ids_linked), 'argument_linked'] = True
    df_all.argument_linked = df_all.argument_linked.fillna(False)
else:
    df_all['argument_linked'] = False

num_cols_args = ['id_premise', 'id_connection','id_conclusion']
num_cols_locs = ['id_locution', 'id_connection','id_illocution']
df_all[num_cols_args] = df_all[num_cols_args].astype('str')
df_all_loc[num_cols_locs] = df_all_loc[num_cols_locs].astype('str')

df_1 = df_all.merge(df_all_loc[['locution', 'id_illocution']], left_on = 'id_conclusion', right_on = 'id_illocution', how='left')
df_1 = df_1.iloc[:, :-1]
df_1.columns = ['premise', 'connection', 'conclusion', 'id_premise', 'id_connection',
                'id_conclusion', 'nodeset_id', 'argument_linked', 'locution_conclusion']

df_2 = df_1.merge(df_all_loc[['locution', 'id_illocution']], left_on = 'id_premise', right_on = 'id_illocution', how='left')
df_2 = df_2.iloc[:, :-1]
df_2.columns = ['premise', 'connection', 'conclusion', 'id_premise', 'id_connection',
                'id_conclusion', 'nodeset_id', 'argument_linked', 'locution_conclusion', 'locution_premise']

df_2 = df_2[['locution_conclusion', 'locution_premise', 'conclusion', 'premise',
             'connection', 'nodeset_id', 'id_conclusion', 'id_premise', 'id_connection', 'argument_linked']]

df_2['speaker_conclusion'] = df_2.locution_conclusion.apply(lambda x: str(str(x).split(' : ')[0]).strip() )
df_2['speaker_premise'] = df_2.locution_premise.apply(lambda x: str(str(x).split(' : ')[0]).strip() )
df_2['speaker'] = df_2.apply(lambda x: x['speaker_conclusion'] in x['speaker_premise'], axis=1)
df_2['speaker'] = np.where(df_2['speaker'] == True, df_2['speaker_conclusion'], '')
df_2['speaker'] = np.where( (df_2['speaker'] == '') & (df_2.id_premise > df_2.id_conclusion) , df_2['speaker_premise'], df_2['speaker'])
df_2['speaker'] = np.where( (df_2['speaker'] == '') & (df_2.id_premise < df_2.id_conclusion) , df_2['speaker_conclusion'], df_2['speaker'])

df_2 = df_2.sort_values(by = ['id_connection', 'id_premise', 'id_conclusion'])

arg_stats = pd.DataFrame(df_2.connection.value_counts().sort_values(ascending=False)).reset_index()
arg_stats.columns = ['Type', 'Number']
arg_stats_prc = pd.DataFrame(df_2.connection.value_counts(normalize=True).round(3).sort_values(ascending=False)*100).reset_index()
arg_stats_prc.columns = ['Type', 'Percentage']
arg_stats = pd.concat( [arg_stats, arg_stats_prc.iloc[:, -1:]], axis=1 )


colors_rels = {
    'Default Inference':'darkblue', 'Default Rephrase':'gold', 'Default Conflict':'darkred',
}

#stats1 = sns.catplot(kind = 'bar', data = arg_stats, x = 'Number', y = 'Type',
                    #height = 4.5, aspect=1.8, palette=colors_rels)
#stats1.set(ylabel='')
#plt.show()
#fig, ax = plt.subplots(figsize = (11, 6))

arg_stats_spk = pd.DataFrame(df_2.speaker.value_counts().sort_values(ascending=False)).reset_index()
arg_stats_spk.columns = ['Speaker', 'Number']
arg_stats_prc_spk = pd.DataFrame(df_2.speaker.value_counts(normalize=True).round(3).sort_values(ascending=False)*100).reset_index()
arg_stats_prc_spk.columns = ['Speaker', 'Percentage']
arg_stats_spk = pd.concat( [arg_stats_spk, arg_stats_prc_spk.iloc[:, -1:]], axis=1 )


sns.set(font_scale=1.65, style='whitegrid')
stats_spk = sns.catplot(kind = 'bar', data = arg_stats_spk, x = 'Percentage', y = 'Speaker',
                    height = 6, aspect=1.8, palette = ['#6F6F6F'])
stats_spk.set(ylabel='', title='Speakers distribution', xticks = np.arange(0, arg_stats_prc_spk.Percentage.max()+11, 10))
plt.show()


sns.set(font_scale=1.45, style='whitegrid')
stats2 = sns.catplot(kind = 'bar', data = arg_stats, x = 'Percentage', y = 'Type',
                    height = 4.5, aspect=1.8, palette=colors_rels)
stats2.set(ylabel='', title='Connection distribution', xticks = np.arange(0, arg_stats_prc.Percentage.max()+11, 20))
plt.show()

col1_stats1, col2_stats1 = st.columns([2, 3], gap = 'small')
with col1_stats1:
    st.write("### Connection Counts")
    add_spacelines(2)
    #st.dataframe(arg_stats.set_index('Type'), width=1000)
    n_metrics = len(arg_stats)
    for nn in range(int(n_metrics)):
        col1_stats1.metric(arg_stats['Type'].iloc[nn], arg_stats['Number'].iloc[nn])
with col2_stats1:
    add_spacelines(2)
    #st.pyplot(stats1)
    st.pyplot(stats2)


st.write('***********************************************************************************************')

col1_stats11, col2_stats11, col3_stats11 = st.columns([1, 1, 3], gap = 'small')
n_metrics_spk = round(len(arg_stats_spk) / 2, 0)
with col1_stats11:
    st.write("### Speakers")
    add_spacelines(2)
    arg_stats_spk1 = arg_stats_spk.iloc[:int(n_metrics_spk)]
    for nn in range(len(arg_stats_spk1)):
        col1_stats11.metric(arg_stats_spk1['Speaker'].iloc[nn], arg_stats_spk1['Number'].iloc[nn])

with col2_stats11:
    st.write("### ")
    add_spacelines(4)
    arg_stats_spk2 = arg_stats_spk.iloc[int(n_metrics_spk):]
    for nn in range(len(arg_stats_spk2)):
        col2_stats11.metric(arg_stats_spk2['Speaker'].iloc[nn], arg_stats_spk2['Number'].iloc[nn])

with col3_stats11:
    add_spacelines(4)
    st.pyplot(stats_spk)


df_2 = df_2.reset_index()
df_2["time"] = df_2['index'].astype('int')
df_2["velocity"] = 1

fig = px.bar(df_2, x="time", y="velocity", hover_data = {"speaker":True, "velocity":False}, color = "speaker",
             labels={'velocity':'', 'speaker': 'Speaker'}, title = "Speakers distribution in time",
             width=950, height=460)
fig.update_layout(xaxis={"tickformat":"d"},
                  font=dict(size=15,color='#000000'),
                   yaxis = dict(tickmode = 'linear',tick0 = 0,dtick = 1
    ))
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig)
df_2 = df_2.drop(columns = ['index', 'time', 'velocity'], axis = 1)


st.write('***********************************************************************************************')


st.write("### Download converted corpora")
col1_download, col2_download = st.columns([2, 3], gap='small')
with col1_download:
    add_spacelines(2)
    download_type = st.radio('Choose file format', ('CSV', 'TSV'))
    #f_extension = str(download_type).replace('Excel', 'xlsx').replace('CSV', 'csv')

    @st.cache
    def convert_df(df, download_type = download_type):
        if download_type == 'CSV':
            return df.to_csv().encode('utf-8')
        else:
            return df.to_csv(sep='\t').encode('utf-8')

    file_download = convert_df(df_2)
    add_spacelines(2)
    st.download_button(
        label="Click to download",
        data=file_download,
        file_name=f'AIF_converted_corpora.csv',
        mime='text/csv',
        )

with col2_download:
    add_spacelines(2)
    df_2.argument_linked = df_2.argument_linked.astype('int')
    #st.dataframe(df_2.iloc[:, 2:].set_index('speaker').head(), width=850)
    st.dataframe(df_2.set_index('speaker').head(), width=850)


#if own_files == 'Nodeset ID from AIF':
    #for f in maps:
        #os.remove(f)

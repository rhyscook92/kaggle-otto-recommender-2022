import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import os
import glob

def get_train(validation, sample_prop=None, columns=None):

    if validation:
        path_to_data = 'validation/train_parquet/'
    else:
        path_to_data = 'test/train_parquet/'

    train = pd.read_parquet(path_to_data, columns=columns, engine='fastparquet')

    if (sample_prop is not None) and (validation):
        unique_sessions = train['session'].unique()
        np.random.seed(42)
        sessions = np.random.choice(
            unique_sessions,
            size = int(sample_prop * len(unique_sessions)),
            replace=False
        )
        reduced_df = train.loc[train['session'].isin(sessions)]
        del train
        gc.collect()
        return(reduced_df)
    else:
        return(train)

def get_test(validation, sample_prop=None, columns=None):
    
    if validation:
        path_to_data = 'validation/test_parquet/'
    else:
        path_to_data = 'test/test_parquet/'
        
    test = pd.read_parquet(path_to_data, columns=columns)
    if (sample_prop is not None) and (validation):
        unique_sessions = test['session'].unique()
        np.random.seed(42)
        sessions = np.random.choice(
          unique_sessions,
          size = int(sample_prop * len(unique_sessions)),
          replace=False
        )
        reduced_df = test.loc[test['session'].isin(sessions)]
        del test
        gc.collect()
        return reduced_df
    else:
        return test
        
def convert_columns(df):
  for column in df.columns:
    if column == 'ts':
      pass
    elif column in ['click_response', 'cart_response', 'order_response']:
      df[column] = df[column].astype('int8')
    elif df[column].dtype == 'float64':
      df[column] = df[column].astype('float32')
    elif df[column].dtype == 'int64':
      df[column] = df[column].astype('int32')
    
  return df
  
def save_parquet(df, directory, files=10, split_column = 'session'):
  ''' saves a parquet in chunks by splitting on a column in the dataframe '''
  splits = df[split_column].unique()
  splits.sort()
  splits_lists = [np_array.tolist() for np_array in np.array_split(splits, files) ]
  
  #Make directory if it doesn't exist
  make_directory(directory)
  
  #Remove files incase parquet already exists
  files = glob.glob(f'{directory}/*')
  for f in files:
    os.remove(f)
    
  #Write the parquet
  for i, split_list in enumerate(tqdm(splits_lists)):
    chunk = df.loc[(df['session'] >= min(split_list)) & (df['session'] <= max(split_list))]
    chunk.to_parquet(f'{directory}/{i}.parquet', index=False)
    del chunk
  return
  
def make_directory(directory):
  if not os.path.exists(directory):
    os.mkdir(directory)
  return
  
def create_sub(click_preds, cart_preds, order_preds):
  ''' Takes subs with sessions and aids and converts to a kaggle submission '''

  make_directory('./submission')
  
  sub = []
  rec_types = ['clicks', 'carts', 'orders']
  rec_dfs = [click_preds, cart_preds, order_preds]
  for rec_type, rec_df in zip(rec_types, rec_dfs):
    assert rec_df.groupby(['session']).agg({'aid' : 'count'})['aid'].min() == rec_df.groupby(['session']).agg({'aid' : 'count'})['aid'].max() == 20
    rec_df['session'] = rec_df['session'].astype(str) + '_' + rec_type
    rec_df = rec_df.groupby('session', as_index=False)['aid'].apply(lambda x: " ".join(map(str, x)))
    sub.append(rec_df)
  sub = pd.concat(sub)
  sub.rename(columns={'session' : 'session_type', 'aid' : 'labels'}, inplace=True)

  assert sub.shape == (5015409, 2)
  assert sub.columns.tolist() == ['session_type', 'labels']
  sub.to_csv('./submission/submission.csv', index=False)

  return
  
def calculate_recall(labels, candidates, recall_type=None):
  candidates = candidates.groupby('session', as_index=False).apply(lambda x : [x.aid.to_list()])
  candidates.columns = ['session', 'candidates']
  test_labels = labels.merge(candidates, on='session', how='inner')
  score = 0
  weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}

  if recall_type is None:
    for t in ['clicks','carts','orders']:
      scored = test_labels.loc[test_labels['type'] == t].copy()
      scored['hits'] = scored.apply(lambda df: len(set(df.ground_truth).intersection(set(df.candidates))), axis=1)
      scored['gt_count'] = scored.ground_truth.str.len().clip(0,20)
      recall = scored['hits'].sum() / scored['gt_count'].sum()
      score += weights[t]*recall
      print(f'{t} recall =',recall)
    print('Overall Recall =',score)
    return score 
      
  if recall_type is not None:
    scored = test_labels.loc[test_labels['type'] == recall_type].copy()
    scored['hits'] = scored.apply(lambda df: len(set(df.ground_truth).intersection(set(df.candidates))), axis=1)
    scored['gt_count'] = scored.ground_truth.str.len().clip(0,20)
    recall = scored['hits'].sum() / scored['gt_count'].sum()
    return recall
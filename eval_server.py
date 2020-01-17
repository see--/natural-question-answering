# adapted from:
# https://www.kaggle.com/c/tensorflow2-question-answering/discussion/120061
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def in_shorts(row):
  return row['short_answer'] in row['short_answers']


def get_f1(answer_df, label_df):
    short_label = (label_df['short_answers'] != '').astype(int)
    long_label = (label_df['long_answer'] != '').astype(int)

    long_predict = np.zeros(answer_df.shape[0])
    long_predict[(answer_df['long_answer'] == label_df['long_answer']) & (answer_df['long_answer'] != '')] = 1
    long_predict[(label_df['long_answer'] == '') & (answer_df['long_answer'] != '')] = 1  # false positive

    short_predict = np.zeros(answer_df.shape[0])
    short_predict[(label_df['short_answers'] == '') & (answer_df['short_answer'] != '')] = 1  # false positive
    a = pd.concat([answer_df[['short_answer']], label_df[['short_answers']]],
        axis=1)
    a['short_answers'] = a['short_answers'].apply(lambda x: x.split())
    short_predict[a.apply(lambda x: in_shorts(x), axis=1) & (a['short_answer'] != '')] = 1

    long_f1 = f1_score(long_label.values, long_predict)
    short_f1 = f1_score(short_label.values, short_predict)
    micro_f1 = f1_score(np.concatenate([long_label, short_label]),
        np.concatenate([long_predict, short_predict]))
    return micro_f1, long_f1, short_f1


def reshape_df(df, short_col):
  short_answers = df[df['example_id'].str.contains('_short')]['PredictionString'].values
  long_answers = df[df['example_id'].str.contains('_long')]['PredictionString'].values
  return pd.DataFrame({short_col: short_answers, 'long_answer': long_answers})


def main(args=None):
  if args is None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='nq-val-v0.8.2.csv')
    parser.add_argument('--fn', type=str, required=True)
    args = parser.parse_args()

  gt = pd.read_csv(args.csv).sort_values('example_id').fillna('')
  pred = pd.read_csv(args.fn).sort_values('example_id').fillna('')
  if gt.shape != pred.shape:
    raise ValueError(f'Your prediction file should have {gt.shape[0]} rows')

  pred_ids = set(pred['example_id'].values)
  for example_id in gt['example_id']:
    if example_id not in pred_ids:
      raise ValueError(f'You have to predict for every row')

  gt = reshape_df(gt, short_col='short_answers')
  pred = reshape_df(pred, short_col='short_answer')
  is_blank = (gt['short_answers'] == '') & (gt['long_answer'] == '')

  micro_f1, long_f1, short_f1 = get_f1(pred, gt)
  micro_f1, long_f1, short_f1 = map(lambda s: round(s, 3),
      (micro_f1, long_f1, short_f1))
  print(f'[ALL - {pred.shape[0]}] Your submission scored: {micro_f1} with {short_f1}'
      f' for short answers and {long_f1} for long answers')

  # micro_f1, long_f1, short_f1 = get_f1(pred[is_blank], gt[is_blank])
  # micro_f1, long_f1, short_f1 = map(lambda s: round(s, 3),
  #     (micro_f1, long_f1, short_f1))
  # print(f'[BLANK] Your submission scored: {micro_f1} with {short_f1}'
  #     f' for short answers and {long_f1} for long answers')

  micro_f1, long_f1, short_f1 = get_f1(pred[~is_blank], gt[~is_blank])
  micro_f1, long_f1, short_f1 = map(lambda s: round(s, 3),
      (micro_f1, long_f1, short_f1))
  print(f'[NON-BLANK - {(~is_blank).sum()}] Your submission scored: {micro_f1} with {short_f1}'
      f' for short answers and {long_f1} for long answers')

  return {'f1': micro_f1, 'short_f1': short_f1, 'long_f1': long_f1}


if __name__ == '__main__':
  main()

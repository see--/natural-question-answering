# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Export the model to [SavedModel format](https://www.tensorflow.org/hub/tf2_saved_model).
   This script is adapted from:
   https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py

  To run this script:
  * 'weights.h5' should be downloaded and unzipped from:
    https://www.kaggle.com/seesee/nq-bert-uncased-68
"""
import argparse
import tensorflow as tf
import os

from transformers import BertConfig, BertTokenizer
from models import TFBertForNaturalQuestionAnswering
from utils_nq import get_add_tokens


def convert_to_functional_api(model):
  input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_type_ids")
  start_logits, end_logits, long_logits = model([input_word_ids, input_mask, input_type_ids])

  return tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                        outputs=[start_logits, end_logits, long_logits],
                        name='tf_bert_for_natural_question_answering')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--hub_destination', type=str, default='bert_uncased_tf2_qa')
  parser.add_argument('--tokenizer_destination', type=str, default='tokenizer_tf2_qa')
  parser.add_argument('--weights_fn', type=str, default='weights.h5')
  args = parser.parse_args()

  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  tokenizer.add_tokens(get_add_tokens(do_enumerate=False), offset=1)
  os.makedirs(args.tokenizer_destination, exist_ok=True)
  tokenizer.save_pretrained(args.tokenizer_destination)
  print(f'Tokenizer is saved to {args.tokenizer_destination}')

  config = BertConfig.from_pretrained('bert-large-uncased')
  model = TFBertForNaturalQuestionAnswering(config)
  _ = model(model.dummy_inputs)
  model.load_weights(args.weights_fn)
  functional_model = convert_to_functional_api(model)
  functional_model.to_lower_case = tf.Variable(True, trainable=False)
  functional_model.save(args.hub_destination, include_optimizer=False, save_format='tf')
  print(f'SavedModel is saved to {args.hub_destination}')


if __name__ == '__main__':
  main()

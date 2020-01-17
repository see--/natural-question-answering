# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (Bert, Roberta)."""
import argparse
import logging
import os
import random
import glob
import time
import pickle
import json
import gc
import math
from shutil import rmtree, copy
from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy as sce

import numpy as np
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa

from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer

import optimization
from utils_nq import read_nq_examples, convert_examples_to_crops, \
    write_predictions, get_add_tokens, convert_preds_to_df, \
    read_candidates
from models import TFBertForNaturalQuestionAnswering, TFRobertaForNaturalQuestionAnswering
from eval_server import main as evaluate_on_nq

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, TFRobertaForNaturalQuestionAnswering, RobertaTokenizer),
}

RawResult = namedtuple("RawResult", ["unique_id", "start_logits", "end_logits",
    "long_logits"])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def train(args, model_class, tokenizer, config, strategy):
    train_dataset, _ = load_and_cache_crops(args, tokenizer, evaluate=False)
    num_replicas = strategy.num_replicas_in_sync
    args.train_batch_size = args.per_tpu_train_batch_size * num_replicas
    num_steps_per_epoch = len(train_dataset['input_ids']) // args.train_batch_size
    t_total = num_steps_per_epoch * args.num_train_epochs
    warmup_steps = int(args.warmup * t_total)

    opt = tf.data.Options()
    opt.experimental_deterministic = True
    train_ds = tf.data.Dataset.from_tensor_slices(train_dataset).with_options(opt)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(buffer_size=100, seed=args.seed)
    train_ds = train_ds.batch(batch_size=args.train_batch_size, drop_remainder=True)
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    train_ds = iter(train_ds)

    with strategy.scope():
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        if args.init_weights:
            logger.info(f"Initializing from {args.init_weights}")
            model.load_weights(args.init_weights)

        optimizer = optimization.create_optimizer(args.learning_rate,
            num_steps_per_epoch * args.num_train_epochs, warmup_steps)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    print(model.summary())

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            start_loss = sce(batch['start_positions'], outputs[0], from_logits=True)
            end_loss = sce(batch['end_positions'], outputs[1], from_logits=True)
            long_loss = sce(batch['long_positions'], outputs[2], from_logits=True)
            loss = ((tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss) / 2.0) +
                tf.reduce_mean(long_loss)) / 2.0
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset["input_ids"]))
    logger.info("  Num Epochs = %.1f", args.num_train_epochs)
    logger.info("  Instantaneous batch size per TPU = %d", args.per_tpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    num_samples = 0
    smooth = 0.99
    set_seed(args)
    for epoch in range(1, math.ceil(args.num_train_epochs) + 1):
        running_loss = 0.0
        epoch_iterator = tqdm(range(num_steps_per_epoch))
        for step in epoch_iterator:
            if global_step >= t_total:
                logger.info(f'Finished training at epoch {epoch}, step {global_step}')
                break
            batch = next(train_ds)
            loss = strategy.experimental_run_v2(train_step, args=(batch, ))
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
            global_step += 1
            running_loss = smooth * running_loss + (1. - smooth) * float(loss)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                step_str = '%06d' % global_step
                ckpt_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(step_str))
                os.makedirs(ckpt_dir, exist_ok=True)
                weights_fn = os.path.join(ckpt_dir, 'weights.h5')
                model.save_weights(weights_fn)
                tokenizer.save_pretrained(ckpt_dir)

                # remove too many checkpoints
                checkpoint_fns = sorted(glob.glob(os.path.join(args.output_dir, 'checkpoint-*')))
                for fn in checkpoint_fns[:-args.n_keep]:
                    rmtree(fn)

            epoch_iterator.set_postfix({'ep': '%d/%.1f' % (epoch, round(args.num_train_epochs, 1)),
                'samples': num_samples, 'trl': round(running_loss, 4)})
            num_samples += args.train_batch_size

    return global_step, running_loss


def evaluate(args, model, tokenizer, strategy):
    val = 'val' in args.predict_fn
    prefix = 'val' if val else 'test'
    raw_results_fn = os.path.join(args.output_dir, f'{prefix}_raw_results.pkl')
    prediction_fn = os.path.join(args.output_dir, f'{prefix}_predictions.json')
    nbest_fn = os.path.join(args.output_dir, f'{prefix}_nbest_predictions.json')
    csv_fn = os.path.join(args.output_dir, f'{prefix}_submission.csv')
    null_log_odds_fn = os.path.join(args.output_dir, f'{prefix}_null_odds.json')
    eval_dataset, crops = load_and_cache_crops(args, tokenizer, evaluate=True)
    if not os.path.exists(raw_results_fn):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        num_replicas = strategy.num_replicas_in_sync
        args.eval_batch_size = args.per_tpu_eval_batch_size * num_replicas

        # pad dataset to multiple of `args.eval_batch_size`
        eval_dataset_length = len(eval_dataset['input_ids'])
        padded_length = math.ceil(eval_dataset_length / args.eval_batch_size) * args.eval_batch_size
        num_pad = padded_length - eval_dataset_length
        for ti, t in eval_dataset.items():
            pad_tensor = tf.expand_dims(tf.zeros_like(t[0]), 0)
            pad_tensor = tf.repeat(pad_tensor, num_pad, 0)
            eval_dataset[ti] = tf.concat([t, pad_tensor], 0)

        # create eval dataset
        eval_dataset['example_index'] = tf.range(padded_length, dtype=tf.int32)
        eval_ds = tf.data.Dataset.from_tensor_slices(eval_dataset)
        eval_ds = eval_ds.batch(batch_size=args.eval_batch_size, drop_remainder=True)
        eval_ds = strategy.experimental_distribute_dataset(eval_ds)
        eval_ds = iter(eval_ds)

        # eval
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", eval_dataset_length)
        logger.info("  Batch size = %d", args.eval_batch_size)

        @tf.function
        def predict_step(batch):
            outputs = model(batch, training=False)
            return outputs

        all_results = []
        tic = time.time()
        for batch_ind in tqdm(range(padded_length // args.eval_batch_size)):
            # if batch_ind > 40:
            #     break
            batch = next(eval_ds)
            example_indexes = tf.concat(batch['example_index'].values, 0)
            outputs = strategy.experimental_run_v2(predict_step, args=(batch, ))
            outputs = [tf.concat(o.values, 0).numpy() for o in outputs]

            for i, example_index in enumerate(example_indexes):
                # filter out padded samples
                if example_index >= eval_dataset_length:
                    continue

                eval_crop = crops[example_index]
                unique_id = int(eval_crop.unique_id)
                start_logits = outputs[0][i]
                end_logits = outputs[1][i]
                long_logits = outputs[2][i]

                start_logits = [float(x) for x in start_logits]
                end_logits = [float(x) for x in end_logits]
                long_logits = [float(x) for x in long_logits]

                result = RawResult(unique_id=unique_id,
                                   start_logits=start_logits,
                                   end_logits=end_logits,
                                   long_logits=long_logits)
                all_results.append(result)

        eval_time = time.time() - tic
        logger.info("  Evaluation done in total %f secs (%f sec per example)",
            eval_time, eval_time / padded_length)
        with open(raw_results_fn, 'wb') as f:
            pickle.dump(all_results, f)
        print(f'***** Done wrote raw results to {raw_results_fn} *****')

    else:
        print(f'***** Loading raw results from {raw_results_fn} *****')
        with open(raw_results_fn, 'rb') as f:
          all_results = pickle.load(f)

    examples_gen = read_nq_examples(args.predict_fn, is_training=False)
    preds = write_predictions(examples_gen, crops, all_results, args.n_best_size,
                              args.max_answer_length,
                              prediction_fn, nbest_fn, null_log_odds_fn,
                              args.verbose_logging,
                              args.short_null_score_diff_threshold, args.long_null_score_diff_threshold)
    del crops, all_results
    gc.collect()
    candidates = read_candidates(['simplified-nq-train.jsonl', 'simplified-nq-test.jsonl'])
    sub = convert_preds_to_df(preds, candidates).sort_values('example_id')
    sub.to_csv(csv_fn, index=False, columns=['example_id', 'PredictionString'])
    print(f'***** Wrote submission to {csv_fn} *****')
    result = {}
    if val:
        EvalArgs = namedtuple('EvalArgs', ['csv', 'fn'])
        eval_args = EvalArgs(csv=args.gt_csv, fn=csv_fn)
        result = evaluate_on_nq(eval_args)
    return result


def load_and_cache_crops(args, tokenizer, evaluate=False):
    # Load data crops from cache or dataset file
    input_file = args.predict_fn if evaluate else args.train_fn
    cache_prefix = 'train'
    if evaluate:
        if 'test' in input_file:
            cache_prefix = 'test'
        else:
            cache_prefix = 'val'

    cached_crops_fn = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}.pkl'.format(
        cache_prefix, list(filter(None, args.output_dir.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_crops_fn) and not args.overwrite_cache:
        logger.info("Loading crops from cached file %s", cached_crops_fn)
        with open(cached_crops_fn, "rb") as f:
            crops = pickle.load(f)
    else:
        if args.model_name_or_path.startswith('bert'):
          cls_token = '[CLS]'
          sep_token = '[SEP]'
          pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
          sep_token_extra = False
        elif args.model_name_or_path.startswith('roberta'):
          cls_token = '<s>'
          sep_token = '</s>'
          pad_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
          sep_token_extra = True

        doc_stride = args.eval_doc_stride if evaluate else args.train_doc_stride
        logger.info(f"Creating crops of length {args.max_seq_length} with stride {doc_stride}"
            f" from dataset file {input_file} using cls_token {cls_token}, sep_token {sep_token}"
            f" and pad_id {pad_id}")

        examples_gen = read_nq_examples(input_file, is_training=not evaluate)
        crops = convert_examples_to_crops(examples_gen=examples_gen,
                                          tokenizer=tokenizer,
                                          max_seq_length=args.max_seq_length,
                                          doc_stride=doc_stride,
                                          max_query_length=args.max_query_length,
                                          is_training=not evaluate,
                                          cls_token_segment_id=0,
                                          pad_token_segment_id=0,
                                          cls_token=cls_token,
                                          sep_token=sep_token,
                                          pad_id=pad_id,
                                          p_keep_impossible=args.p_keep_impossible if not evaluate else 1.0,
                                          sep_token_extra=sep_token_extra)
        logger.info("Saving crops into cached file %s", cached_crops_fn)
        with open(cached_crops_fn, "wb") as f:
            pickle.dump(crops, f)

    # stack
    all_input_ids = tf.stack([c.input_ids for c in crops], 0)
    all_attention_mask = tf.stack([c.attention_mask for c in crops], 0)
    all_token_type_ids = tf.stack([c.token_type_ids for c in crops], 0)
    # all_p_mask = tf.stack([c.p_mask for c in crops], 0)
    all_attention_mask = tf.cast(all_attention_mask, tf.int32)

    if evaluate:
        dataset = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids
        }

    else:
        all_start_positions = tf.convert_to_tensor([f.start_position for f in crops], dtype=tf.int32)
        all_end_positions = tf.convert_to_tensor([f.end_position for f in crops], dtype=tf.int32)
        all_long_positions = tf.convert_to_tensor([f.long_position for f in crops], dtype=tf.int32)
        dataset = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids,
            'start_positions': all_start_positions,
            'end_positions': all_end_positions,
            'long_positions': all_long_positions,
        }

    # https://github.com/huggingface/transformers/blob/master/examples/run_squad.py#L206
    if args.model_name_or_path.startswith('roberta'):
      logger.info(f"Removing token_type_ids for {args.model_name_or_path}")
      dataset.pop('token_type_ids')

    return dataset, crops


def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError as e:
        print(e)
        print('No TPU detected')
        tpu = None
        strategy = tf.distribute.get_strategy()

    return strategy


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_fn", default="nq-train-v1.0.1.json", type=str)
    parser.add_argument("--predict_fn", default="nq-val-v1.0.1.json", type=str)
    parser.add_argument("--gt_csv", default="nq-val-v1.0.1.csv", type=str)
    parser.add_argument("--model_name_or_path",
        default="bert-large-uncased-whole-word-masking-finetuned-squad", type=str)
    parser.add_argument("--init_weights", default="", type=str)
    parser.add_argument("--id", default="68", type=str)

    # Other parameters
    parser.add_argument('--short_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument('--long_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--train_doc_stride", default=256, type=int)
    parser.add_argument("--eval_doc_stride", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--do_not_train", action='store_true')
    parser.add_argument("--do_not_eval", action='store_true')
    parser.add_argument("--per_tpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_tpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=2.0, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--n_best_size", default=10, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--logging_steps', type=int, default=1_000_000)
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_keep', type=int, default=2, help="Number of checkpoints to keep.")
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.1, help="The fraction of impossible"
                        " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(args)
    if args.model_name_or_path.startswith('bert'):
        do_lower_case = 'uncased' in args.model_name_or_path
    elif args.model_name_or_path.startswith('roberta'):
        # https://github.com/huggingface/transformers/pull/1386/files
        do_lower_case = False

    args.output_dir = f"nq_{args.model_name_or_path.split('-')[0]}_" \
        f"{'uncased' if do_lower_case else 'cased'}_{args.id}"
    strategy = get_strategy()
    logger.info("Running on %s replicas" % strategy.num_replicas_in_sync)
    logger.info("Training / evaluation parameters %s", args)
    args.model_type = args.model_name_or_path.split('-')[0].lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, cache_dir='transformers_cache')
    logger.info(f"Using {'uncased' if do_lower_case else 'cased'} tokenizer")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
        do_lower_case=do_lower_case, cache_dir='transformers_cache')
    add_tokens = get_add_tokens(do_enumerate=args.do_enumerate)
    if args.model_name_or_path.startswith('bert'):
      offset = 1
    elif args.model_name_or_path.startswith('roberta'):
      offset = 50100
      # https://github.com/huggingface/transformers/issues/1234
      # config.type_vocab_size = 2

    num_added = tokenizer.add_tokens(add_tokens, offset=offset)
    logger.info(f"Added {num_added} tokens")
    # Training
    if not args.do_not_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save the code
        src_fns = glob.glob("*.py")
        for src_fn in src_fns:
            dst = os.path.join(args.output_dir, src_fn)
            copy(src_fn, dst)

        # Save the run arguments
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            args_ = vars(args).copy()
            args_['device'] = ''
            json.dump(args_, f, indent=2)

        global_step, tr_loss = train(args, model_class, tokenizer, config, strategy)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if not args.do_not_eval:
        # Reload the model
        glob_str = os.path.join(args.output_dir, 'checkpoint-*')
        checkpoint_fns = sorted(glob.glob(glob_str))
        if len(checkpoint_fns) == 0:
          raise ValueError(f'No weights for {glob_str}')

        checkpoint = checkpoint_fns[-1]
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        weights_fn = os.path.join(checkpoint, 'weights.h5')
        with strategy.scope():
            model = model_class(config)
            model(model.dummy_inputs, training=False)
            model.load_weights(weights_fn)

        # Evaluate
        result = evaluate(args, model, tokenizer, strategy)
        logger.info("Result: {}".format(result))


if __name__ == "__main__":
    main()

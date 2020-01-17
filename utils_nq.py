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
""" Load NQ dataset. """
import json
import logging
import os
import collections
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

from transformers.tokenization_bert import whitespace_tokenize


logger = logging.getLogger(__name__)


NQExample = collections.namedtuple("NQExample", [
    "qas_id", "question_text", "doc_tokens", "orig_answer_text",
    "start_position", "end_position", "long_position",
    "short_is_impossible", "long_is_impossible", "crop_start"])

Crop = collections.namedtuple("Crop", ["unique_id", "example_index", "doc_span_index",
    "tokens", "token_to_orig_map", "token_is_max_context",
    "input_ids", "attention_mask", "token_type_ids",
    # "p_mask",
    "paragraph_len", "start_position", "end_position", "long_position",
    "short_is_impossible", "long_is_impossible"])

LongAnswerCandidate = collections.namedtuple('LongAnswerCandidate', [
    'start_token', 'end_token', 'top_level'])

UNMAPPED = -123
CLS_INDEX = 0


def get_add_tokens(do_enumerate):
    tags = ['Dd', 'Dl', 'Dt', 'H1', 'H2', 'H3', 'Li', 'Ol', 'P', 'Table', 'Td', 'Th', 'Tr', 'Ul']
    opening_tags = [f'<{tag}>' for tag in tags]
    closing_tags = [f'</{tag}>' for tag in tags]
    added_tags = opening_tags + closing_tags
    # See `nq_to_sqaud.py` for special-tokens
    special_tokens = ['<P>', '<Table>']
    if do_enumerate:
        for special_token in special_tokens:
            for j in range(11):
              added_tags.append(f'<{special_token[1: -1]}{j}>')

    add_tokens = ['Td_colspan', 'Th_colspan', '``', '\'\'', '--']
    add_tokens = add_tokens + added_tags
    return add_tokens


def find_closing_tag(tokens, opening_tag):
    closing_tag = f'</{opening_tag[1: -1]}>'
    index, stack = -1, []
    for token_index, token in enumerate(tokens):
        if token == opening_tag:
            stack.insert(0, opening_tag)
        elif token == closing_tag:
            stack.pop()

        if len(stack) == 0:
            index = token_index
            break
    return index


def read_candidates(candidate_files, do_cache=True):
    assert isinstance(candidate_files, (tuple, list)), candidate_files
    for fn in candidate_files:
        assert os.path.exists(fn), f'Missing file {fn}'
    cache_fn = 'candidates.pkl'

    candidates = {}
    if not os.path.exists(cache_fn):
        for fn in candidate_files:
            with open(fn) as f:
                for line in tqdm(f):
                    entry = json.loads(line)
                    example_id = str(entry['example_id'])
                    cnds = entry.pop('long_answer_candidates')
                    cnds = [LongAnswerCandidate(c['start_token'], c['end_token'],
                            c['top_level']) for c in cnds]
                    candidates[example_id] = cnds

        if do_cache:
            with open(cache_fn, 'wb') as f:
                pickle.dump(candidates, f)
    else:
        print(f'Loading from cache: {cache_fn}')
        with open(cache_fn, 'rb') as f:
            candidates = pickle.load(f)

    return candidates


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_nq_examples(input_file_or_data, is_training):
    """Read a NQ json file into a list of NQExample. Refer to `nq_to_squad.py`
       to convert the `simplified-nq-t*.jsonl` files to NQ json."""
    if isinstance(input_file_or_data, str):
        with open(input_file_or_data, "r", encoding='utf-8') as f:
            input_data = json.load(f)["data"]

    else:
        input_data = input_file_or_data

    for entry_index, entry in enumerate(tqdm(input_data, total=len(input_data))):
        # if entry_index >= 2:
        #     break
        assert len(entry["paragraphs"]) == 1
        paragraph = entry["paragraphs"][0]
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        assert len(paragraph["qas"]) == 1
        qa = paragraph["qas"][0]
        start_position = None
        end_position = None
        long_position = None
        orig_answer_text = None
        short_is_impossible = False
        long_is_impossible = False
        if is_training:
            short_is_impossible = qa["short_is_impossible"]
            short_answers = qa["short_answers"]
            if len(short_answers) >= 2:
                # logger.info(f"Choosing leftmost of "
                #     f"{len(short_answers)} short answer")
                short_answers = sorted(short_answers, key=lambda sa: sa["answer_start"])
                short_answers = short_answers[0: 1]

            if not short_is_impossible:
                answer = short_answers[0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[
                    answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly
                # recovered from the document. If this CAN'T
                # happen it's likely due to weird Unicode stuff
                # so we will just skip the example.
                #
                # Note that this means for training mode, every
                # example is NOT guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position:
                    end_position + 1])
                cleaned_answer_text = " ".join(
                    whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning(
                        "Could not find answer: '%s' vs. '%s'",
                        actual_text, cleaned_answer_text)
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

            long_is_impossible = qa["long_is_impossible"]
            long_answers = qa["long_answers"]
            if (len(long_answers) != 1) and not long_is_impossible:
                raise ValueError(f"For training, each question"
                    f" should have exactly 1 long answer.")

            if not long_is_impossible:
                long_answer = long_answers[0]
                long_answer_offset = long_answer["answer_start"]
                long_position = char_to_word_offset[long_answer_offset]
            else:
                long_position = -1

            # print(f'Q:{question_text}')
            # print(f'A:{start_position}, {end_position},
            # {orig_answer_text}')
            # print(f'R:{doc_tokens[start_position: end_position]}')

            if not short_is_impossible and not long_is_impossible:
                assert long_position <= start_position

            if not short_is_impossible and long_is_impossible:
                assert False, f'Invalid pair short, long pair'

        example = NQExample(
            qas_id=qa["id"],
            question_text=qa["question"],
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            long_position=long_position,
            short_is_impossible=short_is_impossible,
            long_is_impossible=long_is_impossible,
            crop_start=qa["crop_start"])

        yield example


DocSpan = collections.namedtuple("DocSpan", ["start", "length"])


def get_spans(doc_stride, max_tokens_for_doc, max_len):
    doc_spans = []
    start_offset = 0
    while start_offset < max_len:
        length = max_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == max_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def convert_examples_to_crops(examples_gen, tokenizer, max_seq_length,
                              doc_stride, max_query_length, is_training,
                              cls_token='[CLS]', sep_token='[SEP]', pad_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              mask_padding_with_zero=True,
                              p_keep_impossible=None,
                              sep_token_extra=False):
    """Loads a data file into a list of `InputBatch`s."""
    assert p_keep_impossible is not None, '`p_keep_impossible` is required'
    unique_id = 1000000000
    num_short_pos, num_short_neg = 0, 0
    num_long_pos, num_long_neg = 0, 0
    sub_token_cache = {}
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    crops = []
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info('Converting %s: short_pos %s short_neg %s'
                ' long_pos %s long_neg %s',
                example_index, num_short_pos, num_short_neg,
                num_long_pos, num_long_neg)

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # this takes the longest!
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = sub_token_cache.get(token)
            if sub_tokens is None:
                sub_tokens = tokenizer.tokenize(token)
                sub_token_cache[token] = sub_tokens
            tok_to_orig_index.extend([i for _ in range(len(sub_tokens))])
            all_doc_tokens.extend(sub_tokens)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.short_is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.short_is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[
                    example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

        tok_long_position = None
        if is_training and example.long_is_impossible:
            tok_long_position = -1

        if is_training and not example.long_is_impossible:
            tok_long_position = orig_to_tok_index[example.long_position]

        # For Bert: [CLS] question [SEP] paragraph [SEP]
        special_tokens_count = 3
        if sep_token_extra:
            # For Roberta: <s> question </s> </s> paragraph </s>
            special_tokens_count += 1
        max_tokens_for_doc = max_seq_length - len(query_tokens) - special_tokens_count
        assert max_tokens_for_doc > 0
        # We can have documents that are longer than the maximum
        # sequence length. To deal with this we do a sliding window
        # approach, where we take chunks of the up to our max length
        # with a stride of `doc_stride`.
        doc_spans = get_spans(doc_stride, max_tokens_for_doc, len(all_doc_tokens))
        for doc_span_index, doc_span in enumerate(doc_spans):
            # Tokens are constructed as: CLS Query SEP Paragraph SEP
            tokens = []
            token_to_orig_map = UNMAPPED * np.ones((max_seq_length, ), dtype=np.int32)
            token_is_max_context = np.zeros((max_seq_length, ), dtype=np.bool)
            token_type_ids = []

            # p_mask: mask with 1 for token than cannot be in the
            # answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token
            # (set to 0) (not sure why...)
            # p_mask = []

            short_is_impossible = example.short_is_impossible
            start_position = None
            end_position = None
            special_tokens_offset = special_tokens_count - 1
            doc_offset = len(query_tokens) + special_tokens_offset
            if is_training and not short_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    start_position = 0
                    end_position = 0
                    short_is_impossible = True
                else:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            long_is_impossible = example.long_is_impossible
            long_position = None
            if is_training and not long_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                # out of span
                if not (tok_long_position >= doc_start and tok_long_position <= doc_end):
                    long_position = 0
                    long_is_impossible = True
                else:
                    long_position = tok_long_position - doc_start + doc_offset

            # drop impossible samples
            if long_is_impossible:
                if np.random.rand() > p_keep_impossible:
                    continue

            # CLS token at the beginning
            tokens.append(cls_token)
            token_type_ids.append(cls_token_segment_id)
            # p_mask.append(0)  # can be answer

            # Query
            tokens += query_tokens
            token_type_ids += [sequence_a_segment_id] * len(query_tokens)
            # p_mask += [1] * len(query_tokens)  # can not be answer

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_a_segment_id)
            # p_mask.append(1)  # can not be answer
            if sep_token_extra:
                tokens.append(sep_token)
                token_type_ids.append(sequence_a_segment_id)
                # p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # We add `example.crop_start` as the original document
                # is already shifted
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index] + example.crop_start

                token_is_max_context[len(tokens)] = check_is_max_context(doc_spans,
                    doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                token_type_ids.append(sequence_b_segment_id)
                # p_mask.append(0)  # can be answer

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_b_segment_id)
            # p_mask.append(1)  # can not be answer

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_id)
                attention_mask.append(0 if mask_padding_with_zero else 1)
                token_type_ids.append(pad_token_segment_id)
                # p_mask.append(1)  # can not be answer

            # reduce memory, only input_ids needs more bits
            input_ids = np.array(input_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.bool)
            token_type_ids = np.array(token_type_ids, dtype=np.uint8)
            # p_mask = np.array(p_mask, dtype=np.bool)

            if is_training and short_is_impossible:
                start_position = CLS_INDEX
                end_position = CLS_INDEX

            if is_training and long_is_impossible:
                long_position = CLS_INDEX

            if example_index in (0, 10):
                # too spammy otherwise
                if doc_span_index in (0, 5):
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    # logger.info("token_to_orig_map: %s" % " ".join([
                    #     "%d:%d" % (x, y) for (x, y) in enumerate(token_to_orig_map)]))
                    # logger.info("token_is_max_context: %s" % " ".join([
                    #     "%d:%s" % (x, y) for (x, y) in enumerate(token_is_max_context)
                    # ]))
                    logger.info("input_ids: %s" % input_ids)
                    logger.info("attention_mask: %s" % np.uint8(attention_mask))
                    logger.info("token_type_ids: %s" % token_type_ids)
                    if is_training and short_is_impossible:
                        logger.info("short impossible example")
                    if is_training and long_is_impossible:
                        logger.info("long impossible example")
                    if is_training and not short_is_impossible:
                        answer_text = " ".join(tokens[start_position: end_position + 1])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info("answer: %s" % (answer_text))

            if short_is_impossible:
                num_short_neg += 1
            else:
                num_short_pos += 1

            if long_is_impossible:
                num_long_neg += 1
            else:
                num_long_pos += 1

            crop = Crop(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                # p_mask=p_mask,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                long_position=long_position,
                short_is_impossible=short_is_impossible,
                long_is_impossible=long_is_impossible)
            crops.append(crop)
            unique_id += 1

    return crops


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


PrelimPrediction = collections.namedtuple("PrelimPrediction",
    ["crop_index", "start_index", "end_index", "start_logit", "end_logit"])

NbestPrediction = collections.namedtuple("NbestPrediction", [
    "text", "start_logit", "end_logit",
    "start_index", "end_index",
    "orig_doc_start", "orig_doc_end", "crop_index"])


def clean_text(tok_text):
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def get_nbest(prelim_predictions, crops, example, n_best_size):
    seen, nbest = set(), []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        crop = crops[pred.crop_index]
        orig_doc_start, orig_doc_end = -1, -1
        # non-null
        orig_doc_start, orig_doc_end = -1, -1
        if pred.start_index > 0:
            # Long answer has no end_index. We still generate some text to check
            if pred.end_index == -1:
                tok_tokens = crop.tokens[pred.start_index: pred.start_index + 11]
            else:
                tok_tokens = crop.tokens[pred.start_index: pred.end_index + 1]
            tok_text = " ".join(tok_tokens)
            tok_text = clean_text(tok_text)

            orig_doc_start = int(crop.token_to_orig_map[pred.start_index])
            if pred.end_index == -1:
                orig_doc_end = orig_doc_start + 10
            else:
                orig_doc_end = int(crop.token_to_orig_map[pred.end_index])

            final_text = tok_text
            if final_text in seen:
                continue

        else:
            final_text = ""

        seen.add(final_text)
        nbest.append(NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred.start_index, end_index=pred.end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            crop_index=pred.crop_index))

    # Degenerate case. I never saw this happen.
    if len(nbest) in (0, 1):
        nbest.insert(0, NbestPrediction(text="empty",
            start_logit=0.0, end_logit=0.0,
            start_index=-1, end_index=-1,
            orig_doc_start=-1, orig_doc_end=-1,
            crop_index=UNMAPPED))

    assert len(nbest) >= 1
    return nbest


def write_predictions(examples_gen, all_crops, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      short_null_score_diff, long_null_score_diff):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    # create indexes
    example_index_to_crops = collections.defaultdict(list)
    for crop in all_crops:
        example_index_to_crops[crop.example_index].append(crop)
    unique_id_to_result = {result.unique_id: result for result in all_results}

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    short_num_empty, long_num_empty = 0, 0
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info(f'[{example_index}]: {short_num_empty} short and {long_num_empty} long empty')

        crops = example_index_to_crops[example_index]
        short_prelim_predictions, long_prelim_predictions = [], []
        for crop_index, crop in enumerate(crops):
            assert crop.unique_id in unique_id_to_result, f"{crop.unique_id}"
            result = unique_id_to_result[crop.unique_id]
            # get the `n_best_size` largest indexes
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array#23734295
            start_indexes = np.argpartition(result.start_logits, -n_best_size)[-n_best_size:]
            start_indexes = [int(x) for x in start_indexes]
            end_indexes = np.argpartition(result.end_logits, -n_best_size)[-n_best_size:]
            end_indexes = [int(x) for x in end_indexes]

            # create short answers
            for start_index in start_indexes:
                if start_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[start_index] == UNMAPPED:
                    continue
                if not crop.token_is_max_context[start_index]:
                    continue

                for end_index in end_indexes:
                    if end_index >= len(crop.tokens):
                        continue
                    if crop.token_to_orig_map[end_index] == UNMAPPED:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    short_prelim_predictions.append(PrelimPrediction(
                        crop_index=crop_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

            long_indexes = np.argpartition(result.long_logits, -n_best_size)[-n_best_size:].tolist()
            for long_index in long_indexes:
                if long_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[long_index] == UNMAPPED:
                    continue
                # TODO(see--): Is this needed?
                # -> Yep helps both short and long by about 0.1
                if not crop.token_is_max_context[long_index]:
                    continue
                long_prelim_predictions.append(PrelimPrediction(
                    crop_index=crop_index,
                    start_index=long_index, end_index=-1,
                    start_logit=result.long_logits[long_index],
                    end_logit=result.long_logits[long_index]))

        short_prelim_predictions = sorted(short_prelim_predictions,
            key=lambda x: x.start_logit + x.end_logit, reverse=True)

        short_nbest = get_nbest(short_prelim_predictions, crops,
            example, n_best_size)

        short_best_non_null = None
        for entry in short_nbest:
            if short_best_non_null is None:
                if entry.text != "":
                    short_best_non_null = entry

        long_prelim_predictions = sorted(long_prelim_predictions,
            key=lambda x: x.start_logit, reverse=True)

        long_nbest = get_nbest(long_prelim_predictions, crops,
            example, n_best_size)

        long_best_non_null = None
        for entry in long_nbest:
            if long_best_non_null is None:
                if entry.text != "":
                    long_best_non_null = entry

        nbest_json = {'short': [], 'long': []}
        for kk, entries in [('short', short_nbest), ('long', long_nbest)]:
            for i, entry in enumerate(entries):
                output = {}
                output["text"] = entry.text
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                output["orig_doc_start"] = entry.orig_doc_start
                output["orig_doc_end"] = entry.orig_doc_end
                nbest_json[kk].append(output)

        assert len(nbest_json['short']) >= 1
        assert len(nbest_json['long']) >= 1

        # We use the [CLS] score of the crop that has the maximum positive score
        # long_score_diff = min_long_score_null - long_best_non_null.start_logit
        # Predict "" if null score - the score of best non-null > threshold
        try:
            crop_unique_id = crops[short_best_non_null.crop_index].unique_id
            start_score_null = unique_id_to_result[crop_unique_id].start_logits[CLS_INDEX]
            end_score_null = unique_id_to_result[crop_unique_id].end_logits[CLS_INDEX]
            short_score_null = start_score_null + end_score_null
            short_score_diff = short_score_null - (short_best_non_null.start_logit +
                short_best_non_null.end_logit)

            if short_score_diff > short_null_score_diff:
                final_pred = ("", -1, -1)
                short_num_empty += 1
            else:
                final_pred = (short_best_non_null.text, short_best_non_null.orig_doc_start,
                    short_best_non_null.orig_doc_end)
        except Exception as e:
            print(e)
            final_pred = ("", -1, -1)
            short_num_empty += 1

        try:
            long_score_null = unique_id_to_result[crops[
                long_best_non_null.crop_index].unique_id].long_logits[CLS_INDEX]
            long_score_diff = long_score_null - long_best_non_null.start_logit
            scores_diff_json[example.qas_id] = {'short_score_diff': short_score_diff,
                'long_score_diff': long_score_diff}

            if long_score_diff > long_null_score_diff:
                final_pred += ("", -1)
                long_num_empty += 1
                # print(f"LONG EMPTY: {round(long_score_null, 2)} vs "
                #     f"{round(long_best_non_null.start_logit, 2)} (th {long_null_score_diff})")

            else:
                final_pred += (long_best_non_null.text, long_best_non_null.orig_doc_start)

        except Exception as e:
            print(e)
            final_pred += ("", -1)
            long_num_empty += 1

        all_predictions[example.qas_id] = final_pred
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=2))

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=2))

    if output_null_log_odds_file is not None:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=2))

    logger.info(f'{short_num_empty} short and {long_num_empty} long empty of'
        f' {example_index}')
    return all_predictions


def convert_preds_to_df(preds, candidates):
  num_found_long, num_searched_long = 0, 0
  df = {'example_id': [], 'PredictionString': []}
  for example_id, pred in preds.items():
    short_text, start_token, end_token, long_text, long_token = pred
    df['example_id'].append(example_id + '_short')
    short_answer = ''
    if start_token != -1:
      # +1 is required to make the token inclusive
      short_answer = f'{start_token}:{end_token + 1}'
    df['PredictionString'].append(short_answer)

    # print(entry['document_text'].split(' ')[start_token: end_token + 1])
    # find the long answer
    long_answer = ''
    found_long = False
    min_dist = 1_000_000
    if long_token != -1:
      num_searched_long += 1
      for candidate in candidates[example_id]:
        cstart, cend = candidate.start_token, candidate.end_token
        dist = abs(cstart - long_token)
        if dist < min_dist:
          min_dist = dist
        if long_token == cstart:
          long_answer = f'{cstart}:{cend}'
          found_long = True
          break

      if found_long:
        num_found_long += 1
      else:
        logger.info(f"Not found: {min_dist}")

    df['example_id'].append(example_id + '_long')
    df['PredictionString'].append(long_answer)

  df = pd.DataFrame(df)
  print(f'Found {num_found_long} of {num_searched_long} (total {len(preds)})')
  return df

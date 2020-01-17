import tensorflow as tf
from tensorflow.keras import layers as L

from transformers import TFBertMainLayer, TFBertPreTrainedModel, TFRobertaMainLayer, TFRobertaPreTrainedModel
from transformers.modeling_tf_utils import get_initializer


class TFBertForNaturalQuestionAnswering(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name='bert')
        self.initializer = get_initializer(config.initializer_range)
        self.qa_outputs = L.Dense(config.num_labels,
            kernel_initializer=self.initializer, name='qa_outputs')
        self.long_outputs = L.Dense(1, kernel_initializer=self.initializer,
            name='long_outputs')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        long_logits = tf.squeeze(self.long_outputs(sequence_output), -1)
        return start_logits, end_logits, long_logits


class TFRobertaForNaturalQuestionAnswering(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name='roberta')
        self.initializer = get_initializer(config.initializer_range)
        self.qa_outputs = L.Dense(config.num_labels,
            kernel_initializer=self.initializer, name='qa_outputs')
        self.long_outputs = L.Dense(1, kernel_initializer=self.initializer,
            name='long_outputs')

    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        long_logits = tf.squeeze(self.long_outputs(sequence_output), -1)
        return start_logits, end_logits, long_logits


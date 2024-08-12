# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Description of models used for running the algorithm.

Includes base Hard MLP model, and a model combining BP pseudo label with
aggregate embeddings.
"""

import collections

import numpy as np
from pgmax import fgraph  # Assuming you have PGMax installed.
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
import tensorflow as tf


def combined_run_map_bp(
    num_variables: int,
    instance_sims: list[tuple[int, np.ndarray]],
    instance_to_bags: np.ndarray,
    bags: list[tuple[int, int]],
    pairwise_sims: list[tuple[tuple[int, int], float]],
    bag_lambda: float = 4,
    sim_lambda: float = 4,
) -> dict[int, float]:
  """Combines MAP-BP inference with similarity and bag constraints.

  Args:
      num_variables: Number of variables in the factor graph.
      instance_sims: List of instance similarities.
      instance_to_bags: Mapping of instances to bags.
      bags: List of bags (pairs of instances).
      pairwise_sims: List of pairwise similarities.
      bag_lambda: Weight for bag constraints.
      sim_lambda: Weight for similarity constraints.

  Returns:
      Marginal probabilities for each variable.
  """
  print(f'bag_lambda: {bag_lambda}, sim_lambda: {sim_lambda}', flush=True)
  variables = vgroup.NDVarArray(num_states=2, shape=(num_variables,))
  fg = fgraph.FactorGraph(variable_groups=[variables])
  instance_sims_vals = []
  for i in instance_sims:
    instance_sims_vals.append(
        sim_lambda * np.sum(np.array(i[1]), dtype=np.float32)
    )
  instance_sims_vals = np.array(instance_sims_vals, dtype=np.float32)

  log_potential_unaries = (
      -bag_lambda * (1 - 2 * instance_to_bags) - instance_sims_vals
  )
  print(
      f'log_potential_unaries shape: {log_potential_unaries.shape}', flush=True
  )

  unaries = fgroup.EnumFactorGroup(
      variables_for_factors=[[variables[ii]] for ii in range(num_variables)],
      factor_configs=np.arange(2)[:, None],
      log_potentials=np.stack(
          [np.zeros_like(log_potential_unaries), log_potential_unaries], axis=1
      ),
  )

  pair_potentials: dict[tuple[int, int], np.float32] = collections.defaultdict()
  for i in pairwise_sims:
    edge, neighbor = i[0]
    if (
        edge == neighbor
        or (edge, neighbor) in pair_potentials
        or (neighbor, edge) in pair_potentials
    ):
      print('Duplicates:', edge, neighbor, flush=True)
    pair_potentials[(edge, neighbor)] = 2 * sim_lambda * i[1]

  for bag in bags:
    for x1, x2 in bag:
      if (x1, x2) in pair_potentials:
        pair_potentials[(x1, x2)] += -2 * bag_lambda
      else:
        pair_potentials[(x1, x2)] = -2 * bag_lambda

  log_potential_list = np.array(pair_potentials.values(), dtype=np.float32)
  log_potential_matrix = np.zeros(log_potential_list.shape + (2, 2)).reshape(
      (-1, 2, 2)
  )
  log_potential_matrix[:, 1, 1] = log_potential_list
  print(f'log_potential_matrix shape: {log_potential_matrix.shape}', flush=True)

  variables_for_factors = [
      [variables[i[0]], variables[i[1]]] for i in pair_potentials
  ]
  binaries = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_factors,
      log_potential_matrix=log_potential_matrix,
  )

  fg.add_factors([unaries, binaries])
  print('Factor graph created.', flush=True)

  bp = infer.BP(fg.bp_state, temperature=1.0)
  bp_arrays = bp.run(bp.init(), num_iters=100, damping=0.5)
  beliefs = bp.get_beliefs(bp_arrays)
  marginals = infer.get_marginals(beliefs)

  return marginals


class BaseHardMLPModel(tf.keras.Model):
  """Base MLP model with hard decision boundary."""

  def __init__(self):
    super().__init__()
    self.final_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')
    self.hidden_layers = [
        tf.keras.layers.Dense(5040, activation='relu'),
        tf.keras.layers.Dense(1280, activation='relu'),
        tf.keras.layers.Dense(320, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    ]

  def train_step(self, batch):
    inputs, labels = batch
    with tf.GradientTape() as tape:
      preds = self(inputs, training=True)
      loss = self.compiled_loss(labels, preds)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(labels, preds)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, batch):
    inputs, labels = batch
    preds = self(inputs, training=False)
    self.compiled_loss(labels, preds)
    self.compiled_metrics.update_state(labels, preds)
    return {m.name: m.result() for m in self.metrics}

  def call(self, features, training=False):
    out = features
    for hidden_layer in self.hidden_layers:
      out = hidden_layer(out, training=training)
    out = self.final_layer(out, training=training)
    return out

  def get_instance_embeddings(self, inputs, training=False):
    """Extracts instance embeddings from the model."""
    bag_size, feature_size = inputs.shape[1], inputs.shape[2]
    flat_inputs = tf.reshape(inputs, shape=[-1, feature_size])
    instance_embed = flat_inputs
    # Using last hidden layer
    for hidden_layer in self.hidden_layers[:-1]:
      instance_embed = hidden_layer(instance_embed, training=training)
    embed_size = instance_embed.shape[1]
    instance_embed = tf.reshape(instance_embed, [-1, bag_size, embed_size])

    return instance_embed


class BPAggEmbModel(BaseHardMLPModel):
  """Model combining BP aggregation with embeddings."""

  def __init__(self, loss_lambda=0.1):
    super().__init__()
    self.loss_lambda = loss_lambda
    self.bce_loss = tf.keras.losses.BinaryCrossentropy()

    self.bag_embed_layer = tf.keras.layers.Dense(
        units=64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(0.0),
    )
    self.bag_final_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

  def get_instance_embeddings(self, inputs, training=False):
    """Extracts instance embeddings."""
    bag_size, feature_size = inputs.shape[1], inputs.shape[2]
    flat_inputs = tf.reshape(inputs, shape=[-1, feature_size])
    out = flat_inputs
    for hidden_layer in self.hidden_layers[:-1]:
      out = hidden_layer(out, training=training)
    embed_size = tf.shape(out)[-1]
    return tf.reshape(out, [-1, bag_size, embed_size])

  def call(self, features, training=False, mask=None):
    """Forward pass of the model."""

    instance_embed = self.get_instance_embeddings(features, training=training)
    bag_size = features.shape[1]
    preds_size = [-1, bag_size]
    preds = self.final_layer(instance_embed, training=training)
    preds = tf.reshape(preds, preds_size)

    # Pooling the instance embeddings (using mean in this case)
    pooled_embed = tf.reduce_mean(instance_embed, axis=1)

    aggr_embed = self.bag_embed_layer(pooled_embed, training=training)
    aggr_preds = self.bag_final_layer(aggr_embed, training=training)
    aggr_preds = tf.reshape(aggr_preds, [-1, 1])
    return preds, aggr_preds

  def train_step(self, batch):
    """Custom training step for the model."""
    inputs, labels, pseudo_labels = batch
    aggr_labels = tf.reduce_mean(labels, axis=1, keepdims=True)
    with tf.GradientTape() as tape:
      preds, aggr_preds = self(inputs, training=True)
      instance_loss = self.bce_loss(
          tf.reshape(pseudo_labels, (-1,)), tf.reshape(preds, (-1,))
      )
      loss = (
          self.loss_lambda * self.compiled_loss(aggr_labels, aggr_preds)
          + instance_loss
      )
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    if self.compiled_metrics is not None:
      self.compiled_metrics.update_state(
          tf.reshape(labels, (-1,)), tf.reshape(preds, (-1,))
      )
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, batch):
    """Custom test step for the model."""
    inputs, labels = batch
    inputs = tf.expand_dims(inputs, axis=1)
    preds, _ = self(inputs, training=False)
    preds = tf.reshape(preds, (-1,))
    self.compiled_loss(labels, preds)
    self.compiled_metrics.update_state(labels, preds)
    return {m.name: m.result() for m in self.metrics}

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

"""Main control file for the algorithm execution.

Contains all required subfunctions. Starts from scratch loading the data,
processing it, calling the Belief Propagation Pipeline and then iteratinf over
the PseudoLabels and Aggregate Embeddings process. All metrics are logged.
"""

from collections.abc import Sequence
import copy
import os

from absl import app
from absl import flags
from absl import logging
import network
import numpy as np
import pandas as pd
from sklearn import neighbors
import tensorflow as tf
import tensorflow_probability as tfp


_SIM_LAMBDA = flags.DEFINE_float(
    'sim_lambda',
    default=10.0,
    help='tuning for the similarity based covariates',
)

_BAG_LAMBDA = flags.DEFINE_float(
    'bag_lambda', default=10.0, help='tuning for the bag based covariates'
)

_N_NEIGHBORS = flags.DEFINE_integer(
    'n_neighbors', default=5, help='tuning for the number of neighbours'
)

_THRESHOLD = flags.DEFINE_float(
    'threshold',
    default=0.5,
    help='tuning for the threshold to convert from soft to hard labels',
)
_MLP_LR = flags.DEFINE_float(
    'mlp_lr', default=1e-3, help='lr for downstream mlp'
)
_MLP_WD = flags.DEFINE_float('mlp_wd', default=0, help='wd for downstream MLP')
_NEIGH_THRESH = flags.DEFINE_float(
    'neigh_thresh', default=2.0, help='threshold if to include neighs'
)
_Cx = flags.DEFINE_integer(
    'Cx', default=7, help='column x for feature bag constructed'
)
_Cy = flags.DEFINE_integer(
    'Cy', default=10, help='column y for feature bag constructed'
)
_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size',
    default=1024,
    help='No. of instances in a training batch.',
)
_TEST_BATCH_SIZE = flags.DEFINE_integer(
    'test_batch_size', default=2048, help='No. of instances in a testing batch.'
)
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    default='adam',
    enum_values=['sgd', 'adam'],
    help='Optimizer to use for training.',
)
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='/LLP-BP',  # Enter your path here
    help='Experiment CNS directory.',
)
_BAG_SIZE = flags.DEFINE_integer('bag_size', default=8, help='Bag size.')


def load_data_marketing():
  """Loads marketing data."""
  logging.info('Loading marketing data...')

  # Replace with paths in the folder
  x_train_path = '/data/x_train.csv'
  x_test_path = '/data/x_test.csv'
  x_val_path = '/data/x_val.csv'

  y_train_path = 'y_train.csv'
  y_test_path = 'y_test.csv'
  y_val_path = 'y_val.csv'

  x_train_df_itr = pd.read_csv(x_train_path)
  x_test_df_itr = pd.read_csv(x_test_path)
  x_val_df_itr = pd.read_csv(x_val_path)

  y_train_df_itr = pd.read_csv(y_train_path)
  y_test_df_itr = pd.read_csv(y_test_path)
  y_val_df_itr = pd.read_csv(y_val_path)

  n_features = 16

  x_train = x_train_df_itr.iloc[:, :].to_numpy().astype(np.float32)
  y_train = y_train_df_itr.iloc[:, :].to_numpy().astype(np.float32)

  x_train = x_train.reshape((-1, n_features))
  y_train = y_train.reshape((-1, 1))
  x_test = x_test_df_itr.to_numpy().astype(np.float32)
  y_test = y_test_df_itr.to_numpy().astype(np.float32)
  x_val = x_val_df_itr.to_numpy().astype(np.float32)
  y_val = y_val_df_itr.to_numpy().astype(np.float32)

  logging.info('Train data shapes: %s, %s', x_train.shape, y_train.shape)
  pos_frac = np.sum(y_train) / len(y_train)
  logging.info('Positive fraction: %s', pos_frac)

  return x_train, y_train, x_test, y_test, x_val, y_val, n_features


def get_edge_neighbors_pairs(
    kernel, embeds, neighbors_table, n_neighbors=5, thresh=1.0
):
  """Gets example-neighbor pairs based on similarity threshold."""
  logging.info('Getting example-neighbor pairs...')

  all_edge_ids, all_neighbor_ids, all_neighbor_sims = [], [], []

  for i in range(len(neighbors_table[0])):
    edge_id = i

    neighbor_ids = [
        int(neighbors_table[1][i][neighbor])
        for neighbor in range(0, n_neighbors)
    ]
    neighbor_dists = [
        (neighbors_table[0][i][neighbor]) for neighbor in range(0, n_neighbors)
    ]
    count = 0
    for j in range(len(neighbor_ids)):
      if neighbor_dists[j] < thresh and edge_id != neighbor_ids[j]:
        all_edge_ids.append(edge_id)
        all_neighbor_ids.append(neighbor_ids[j])
        all_neighbor_sims.append(
            kernel.apply(x1=embeds[edge_id], x2=embeds[neighbor_ids[j]]).numpy()
        )
        count += 1
        if count == n_neighbors:
          break

  all_edge_ids = np.array(all_edge_ids, dtype=np.int32)
  all_neighbor_ids = np.array(all_neighbor_ids, dtype=np.int32)
  all_neighbor_sims = np.array(all_neighbor_sims, dtype=np.float32)
  return all_edge_ids, all_neighbor_ids, all_neighbor_sims


def get_all_neighbors_acc(all_edge_ids_, all_neighbor_ids_, y_train_):
  """Calculates accuracy based on neighbor labels."""
  logging.info('Calculating accuracy based on neighbor labels...')

  edge_labels = y_train_[all_edge_ids_]
  neighbor_labels = y_train_[all_neighbor_ids_]

  n_corr = np.sum((edge_labels == neighbor_labels).astype('float32'))
  acc = n_corr / len(neighbor_labels)
  logging.info('Accuracy: %s', acc)


def get_knn_acc(all_edge_ids_, all_neighbor_ids_, y_train_):
  """Calculates KNN accuracy."""
  logging.info('Calculating KNN accuracy...')

  edge_labels = y_train_[all_edge_ids_]
  neighbor_labels = y_train_[all_neighbor_ids_]
  aggr_neighbor_labels = (np.mean(neighbor_labels, axis=1) >= 0.5).astype(
      'float32'
  )

  n_corr = np.sum((edge_labels == aggr_neighbor_labels).astype('float32'))
  acc = n_corr / len(edge_labels)
  logging.info('KNN Accuracy: %s', acc)


def train_and_evaluate_map_bp_model(
    x_train,
    y_train,
    bag_size,
    all_edge_ids,
    all_neighbor_ids,
    all_neighbor_sims,
):
  """Trains and evaluates the MAP-BP model."""
  logging.info('Training and evaluating MAP-BP model...')

  # For 1m data
  instance_labels = copy.deepcopy(y_train)
  edge_ids = copy.deepcopy(all_edge_ids)
  neighbor_ids = copy.deepcopy(all_neighbor_ids)
  neighbor_sims = copy.deepcopy(all_neighbor_sims)

  excess_data = len(instance_labels) % bag_size
  logging.info('Excess data: %s', excess_data)
  if excess_data > 0:
    instance_labels = instance_labels[:-excess_data]
    _ = x_train[:-excess_data]

  data_size = len(instance_labels)
  logging.info(
      'Data size: %s, Num pos: %s, Pos fraction: %s',
      data_size,
      np.sum(instance_labels),
      np.sum(instance_labels) / data_size,
  )

  bag_sum_labels = []
  start = 0
  for _ in range(data_size // bag_size):
    label = np.sum(instance_labels[start : start + bag_size])
    start += bag_size
    bag_sum_labels.append(label)

  logging.info('Num bags: %s', len(bag_sum_labels))

  instance_to_bags = []
  for i in range(data_size):
    instance_to_bags.append(bag_sum_labels[i // bag_size])

  bags = []
  start = 0
  for i in range(data_size // bag_size):
    bags.append([])
    for j in range(start, start + bag_size):
      for k in range(j + 1, start + bag_size):
        bags[i].append([j, k])
    start += bag_size

  logging.info('Num bags: %s', len(bags))

  pairwise_sims = []
  all_pairs = set()
  for i, edge_id in enumerate(edge_ids):
    edge = edge_id
    neighbor = neighbor_ids[i]
    if edge < data_size and neighbor < data_size:
      edge_bag_label = bag_sum_labels[edge // bag_size]
      if edge_bag_label <= 1:
        all_pairs.add((edge_id, neighbor))

  for i, edge_id in enumerate(edge_ids):
    edge = edge_id
    neighbor = neighbor_ids[i]
    if edge < data_size and neighbor < data_size:
      bag_label = bag_sum_labels[edge // bag_size]
      if bag_label > 1:
        continue
      if edge > neighbor and (neighbor, edge) in all_pairs:
        continue
      eg = min(edge_ids[i], neighbor_ids[i])
      nbr = max(edge_ids[i], neighbor_ids[i])
      if eg < data_size and nbr < data_size:
        pairwise_sims.append(((eg, nbr), neighbor_sims[i]))

  pairwise_sims = list(set(pairwise_sims))

  logging.info('Num initial neighbor connections: %s', len(all_pairs))
  logging.info('Num unique neighbor connections: %s', len(pairwise_sims))

  num_variables = data_size
  instance_to_bags = np.array(instance_to_bags)

  instance_sims = [[[], []] for _ in range(num_variables)]
  for i in pairwise_sims:
    instance_sims[i[0][0]][0].append(i[0][1])
    instance_sims[i[0][1]][0].append(i[0][0])
    instance_sims[i[0][0]][1].append(i[1])
    instance_sims[i[0][1]][1].append(i[1])

  marginals = network.combined_run_map_bp(
      num_variables,
      instance_sims,
      instance_to_bags,
      bags,
      pairwise_sims,
      bag_lambda=_BAG_LAMBDA.value,
      sim_lambda=_SIM_LAMBDA.value,
  )
  preds = list(marginals.values())[0][:, 1]
  logging.info(
      'Preds sum: %s, Labels sum: %s', np.sum(preds), np.sum(instance_labels)
  )

  auc = tf.keras.metrics.AUC()(instance_labels, preds)
  logging.info('AUC: %s', auc.numpy())

  acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)(instance_labels, preds)
  logging.info('ACC: %s', acc.numpy())

  return marginals, preds, auc, acc


def train_and_evaluate_mlp_model(
    x_train, preds_hard, x_val, y_val, x_test, y_test
):
  """Trains and evaluates the MLP model."""
  logging.info('Training and evaluating MLP model...')

  mlp_model = network.BaseMLPModel()
  optimizer = tf.keras.optimizers.AdamW(
      learning_rate=_MLP_LR.value, weight_decay=_MLP_WD.value
  )
  loss = tf.keras.losses.BinaryCrossentropy()

  metrics_list = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy_0.5', threshold=0.5),
      tf.keras.metrics.BinaryAccuracy(name='accuracy_0.6', threshold=0.6),
      tf.keras.metrics.BinaryAccuracy(name='accuracy_0.7', threshold=0.7),
      tf.keras.metrics.BinaryAccuracy(name='accuracy_0.8', threshold=0.8),
      tf.keras.metrics.Precision(name='precision_0.5', thresholds=0.5),
      tf.keras.metrics.Precision(name='precision_0.6', thresholds=0.6),
      tf.keras.metrics.Precision(name='precision_0.7', thresholds=0.7),
      tf.keras.metrics.Precision(name='precision_0.8', thresholds=0.8),
      tf.keras.metrics.Recall(name='recall_0.5', thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_0.6', thresholds=0.6),
      tf.keras.metrics.Recall(name='recall_0.7', thresholds=0.7),
      tf.keras.metrics.Recall(name='recall_0.8', thresholds=0.8),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.MeanSquaredError(name='mse'),
      tf.keras.metrics.BinaryCrossentropy(name='unweighted_bce'),
      tf.keras.metrics.PrecisionAtRecall(0.5, name='precision_at_recall_0.5'),
      tf.keras.metrics.PrecisionAtRecall(0.6, name='precision_at_recall_0.6'),
      tf.keras.metrics.PrecisionAtRecall(0.7, name='precision_at_recall_0.7'),
      tf.keras.metrics.RecallAtPrecision(0.7, name='recall_at_precision_0.7'),
      tf.keras.metrics.RecallAtPrecision(0.8, name='recall_at_precision_0.8'),
      tf.keras.metrics.RecallAtPrecision(0.9, name='recall_at_precision_0.9'),
  ]

  mlp_model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

  def load_data(train_batch_size, test_batch_size):
    train_data, val_data = (x_train, preds_hard), (x_val, y_val)
    with tf.device('CPU'):
      train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(
          train_batch_size
      )
      val_data = tf.data.Dataset.from_tensor_slices(val_data).batch(
          test_batch_size
      )
    return train_data, val_data

  train_data, val_data = load_data(
      _TRAIN_BATCH_SIZE.value, _TEST_BATCH_SIZE.value
  )

  exp_identifier_str = (
      'bce_'
      f'opt-{_OPTIMIZER.value}_'
      f'bag_size-{_BAG_SIZE.value}_'
      f'lr-{_MLP_LR.value}_'
      f'wd-{_MLP_WD.value}_'
      f'n_neighbors-{_N_NEIGHBORS.value}_'
      f'sim_lambda-{_SIM_LAMBDA.value}_'
      f'bag_lambda-{_BAG_LAMBDA.value}_'
      f'threshold-{_THRESHOLD.value}'
  )
  checkpoint_dir = os.path.join(
      _EXPERIMENT_DIR.value,
      exp_identifier_str,
  )
  log_dir = os.path.join(
      _EXPERIMENT_DIR.value,
      'logs',
      exp_identifier_str,
  )

  callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir=log_dir),
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(
              checkpoint_dir,
              'checkpoint_val_auroc_{epoch:02d}',
          ),
          monitor='val_auc',
          mode='max',
          save_best_only=True,
          save_weights_only=True,
      ),
      tf.keras.callbacks.EarlyStopping(
          monitor='val_auc',
          min_delta=1e-4,
          mode='max',
          patience=4,
          restore_best_weights=True,
          verbose=1,
      ),
  ]

  history = mlp_model.fit(
      train_data, validation_data=val_data, epochs=100, callbacks=callbacks
  )

  mlp_val_auc_best = max(history.history['val_auc'])

  mlp_test_metrics = mlp_model.evaluate(x_test, y_test)
  mlp_test_auc = mlp_test_metrics[17]
  mlp_test_acc = mlp_test_metrics[5]

  mlp_val_metrics = mlp_model.evaluate(x_val, y_val)
  mlp_val_auc = mlp_val_metrics[17]
  mlp_val_acc = mlp_val_metrics[5]

  return (
      mlp_model,
      mlp_val_auc_best,
      mlp_test_auc,
      mlp_test_acc,
      mlp_val_auc,
      mlp_val_acc,
  )


def main(argv: Sequence[str]) -> None:
  """Main execution."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting main execution...')

  # Load data
  x_train, y_train, x_test, y_test, x_val, y_val, n_features = (
      load_data_marketing()
  )
  x_train_og = copy.deepcopy(x_train)
  # KNN model fitting and evaluation
  bag_size = _BAG_SIZE.value
  thresh = _THRESHOLD.value
  matern = tfp.math.psd_kernels.MaternOneHalf()
  kernel = matern
  n_neighbors = _N_NEIGHBORS.value
  knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
  knn.fit(x_train, y_train.reshape((-1,)))
  shuffle_idx = np.arange(x_test.shape[0])
  np.random.shuffle(shuffle_idx)

  x_test = x_test[shuffle_idx]
  y_test = y_test[shuffle_idx]

  neighbors_table = knn.kneighbors()

  # Get example-neighbor pairs
  all_edge_ids, all_neighbor_ids, all_neighbor_sims = get_edge_neighbors_pairs(
      kernel,
      x_train,
      neighbors_table,
      n_neighbors=n_neighbors,
      thresh=_NEIGH_THRESH.value,
  )

  # Train and evaluate MAP-BP model
  _, preds, auc, acc = train_and_evaluate_map_bp_model(
      x_train,
      y_train,
      bag_size,
      all_edge_ids,
      all_neighbor_ids,
      all_neighbor_sims,
  )

  preds_hard = [1 if preds_ >= thresh else 0 for preds_ in preds]
  # Train and evaluate first MLP model
  (
      mlp_model,
      mlp_val_auc_best,
      mlp_test_auc,
      mlp_test_acc,
      mlp_val_auc,
      mlp_val_acc,
  ) = train_and_evaluate_mlp_model(
      x_train, preds_hard, x_val, y_val, x_test, y_test
  )

  bagged_x_train = tf.reshape(x_train_og, [-1, 1, n_features])
  embeds = mlp_model.get_instance_embeddings(bagged_x_train)
  embeds = np.squeeze(embeds, axis=1)

  n_neighbors = _N_NEIGHBORS.value
  knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
  knn.fit(embeds, y_train.reshape((-1,)))

  shuffle_idx = np.arange(x_test.shape[0])
  np.random.shuffle(shuffle_idx)

  x_test = x_test[shuffle_idx]
  y_test = y_test[shuffle_idx]

  test_score = knn.score(
      np.squeeze(
          mlp_model.get_instance_embeddings(
              tf.reshape(x_test, [-1, 1, n_features])
          ),
          axis=1,
      ),
      y_test.reshape((-1,)),
  )
  logging.info('KNN Test Score: %s', test_score)

  val_score = knn.score(
      np.squeeze(
          mlp_model.get_instance_embeddings(
              tf.reshape(x_val, [-1, 1, n_features])
          ),
          axis=1,
      ),
      y_val.reshape((-1,)),
  )
  logging.info('KNN Validation Score: %s', val_score)

  y_pred_proba = knn.predict_proba(
      np.squeeze(
          mlp_model.get_instance_embeddings(
              tf.reshape(x_test, [-1, 1, n_features])
          ),
          axis=1,
      )
  )[:, 1]
  knn_test_auc = tf.keras.metrics.AUC()(y_test, y_pred_proba)

  y_pred_proba_val = knn.predict_proba(
      np.squeeze(
          mlp_model.get_instance_embeddings(
              tf.reshape(x_val, [-1, 1, n_features])
          ),
          axis=1,
      )
  )[:, 1]
  knn_val_auc = tf.keras.metrics.AUC()(y_val, y_pred_proba_val)

  # Second MAP-BP model training and evaluation
  neighbors_table = knn.kneighbors()
  all_edge_ids, all_neighbor_ids, all_neighbor_sims = get_edge_neighbors_pairs(
      kernel,
      embeds,
      neighbors_table,
      n_neighbors=_N_NEIGHBORS.value,
      thresh=_NEIGH_THRESH.value,
  )

  _, preds2, auc2, acc2 = train_and_evaluate_map_bp_model(
      x_train,
      y_train,
      bag_size,
      all_edge_ids,
      all_neighbor_ids,
      all_neighbor_sims,
  )

  # Train and evaluate second MLP model
  thresh = _THRESHOLD.value
  preds_hard2 = [1 if preds_ >= thresh else 0 for preds_ in preds2]
  (
      _,
      mlp2_val_auc_best,
      mlp2_test_auc,
      mlp2_test_acc,
      mlp2_val_auc,
      mlp2_val_acc,
  ) = train_and_evaluate_mlp_model(
      x_train, preds_hard2, x_val, y_val, x_test, y_test
  )

  # Final evaluation and printing
  logging.info('Final Evaluation Results:')
  logging.info('----------------------------')
  logging.info('MAP-BP Model (Initial):')
  logging.info('  - AUC: %s', auc.numpy())
  logging.info('  - Accuracy: %s', acc.numpy())

  logging.info('MLP Model (First):')
  logging.info('  - Best Validation AUC: %s', mlp_val_auc_best)
  logging.info('  - Test AUC: %s', mlp_test_auc)
  logging.info('  - Test Accuracy: %s', mlp_test_acc)
  logging.info('  - Validation AUC: %s', mlp_val_auc)
  logging.info('  - Validation Accuracy: %s', mlp_val_acc)

  logging.info('KNN with Embeddings:')
  logging.info('  - Test Score: %s', test_score)
  logging.info('  - Validation Score: %s', val_score)
  logging.info('  - Test AUC: %s', knn_test_auc)
  logging.info('  - Validation AUC: %s', knn_val_auc)

  logging.info('MAP-BP Model (Second):')
  logging.info('  - AUC: %s', auc2.numpy())
  logging.info('  - Accuracy: %s', acc2.numpy())

  logging.info('MLP Model (Second):')
  logging.info('  - Best Validation AUC: %s', mlp2_val_auc_best)
  logging.info('  - Test AUC: %s', mlp2_test_auc)
  logging.info('  - Test Accuracy: %s', mlp2_test_acc)
  logging.info('  - Validation AUC: %s', mlp2_val_auc)
  logging.info('  - Validation Accuracy: %s', mlp2_val_acc)

  logging.info('Main execution completed.')

  return

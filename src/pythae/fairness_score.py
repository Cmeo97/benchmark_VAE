# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Fairness metric task.

In this task, we compute the unfairness of a classifier trained on a given
representation. This procedure is described in Section 4 of the paper "On the
Fairness of Disentangled Representations", Francesco Locatello, Gabriele Abbati,
Tom Rainforth, Stefan Bauer, Bernhard Schoelkopf, Olivier Bachem, NeurIPS 2019,
https://arxiv.org/abs/1905.13662.
The unfairness of a classifier is computed as the total variation distance
between the distribution of predictions of the target varible for different
interventions on the sensitive variable. This approach is described in Section
4 of the paper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
import sklearn

def compute_fairness(ground_truth_data,
                     representation_function,
                     random_state,
                     num_train=None,
                     num_test_points_per_class=None,
                     batch_size=16):
  """Computes unfairness scores.

  We first compute either the mean or maximum total variation for a given
  sensitive and target variable. Then, we either average or take the maximum
  with respect to target and sensitive variable. For convenience, we compute and
  save all combinations. The score used in Section 4 of the paper is here called
  mean_fairness:mean_pred:mean_sens.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test_points_per_class: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  
  factor_counts = ground_truth_data.factors_num_values
  num_factors = len(factor_counts)

  scores = {}
  # Training a predictive model.
  mus_train, ys_train = generate_batch_factor_code(
      ground_truth_data, representation_function, num_train, random_state,
      batch_size)
  predictor_model_fn = make_predictor_fn()

  # For each factor train a single predictive model.
  mean_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  max_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  for i in range(num_factors):
    model = predictor_model_fn()
    model.fit(np.transpose(mus_train), ys_train[i, :])

    for j in range(num_factors):
      if i == j:
        continue
      # Sample a random set of factors once.
      original_factors = ground_truth_data.sample_factors(
          num_test_points_per_class, random_state)
      counts = np.zeros((factor_counts[i], factor_counts[j]), dtype=np.int64)
      for c in range(factor_counts[j]):
        # Intervene on the sensitive attribute.
        intervened_factors = np.copy(original_factors)
        intervened_factors[:, j] = c
        # Obtain the batched observations.
        observations = ground_truth_data.sample_observations_from_factors(
            intervened_factors, random_state)
        representations = obtain_representation(observations,
                                                      representation_function,
                                                      batch_size)
        # Get the predictions.
        predictions = model.predict(np.transpose(representations))
        # Update the counts.
        counts[:, c] = np.bincount(predictions, minlength=factor_counts[i])
      mean_fairness[i, j], max_fairness[i, j] = inter_group_fairness(counts)

  # Report the scores.
  scores.update(compute_scores_dict(mean_fairness, "mean_fairness"))
  scores.update(compute_scores_dict(max_fairness, "max_fairness"))
  return scores


def compute_scores_dict(metric, prefix):
  """Computes scores for combinations of predictive and sensitive factors.

  Either average or take the maximum with respect to target and sensitive
  variable for all combinations of predictive and sensitive factors.

  Args:
    metric: Matrix of shape [num_factors, num_factors] with fairness scores.
    prefix: Prefix for the matrix in the returned dictionary.

  Returns:
    Dictionary containing all combinations of predictive and sensitive factors.
  """
  result = {}
  # Report min and max scores for each predictive and sensitive factor.
  for i in range(metric.shape[0]):
    for j in range(metric.shape[1]):
      if i != j:
        result["{}:pred{}:sens{}".format(prefix, i, j)] = metric[i, j]

  # Compute mean and max values across rows.
  rows_means = []
  rows_maxs = []
  for i in range(metric.shape[0]):
    relevant_scores = [metric[i, j] for j in range(metric.shape[1]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:pred{}:mean_sens".format(prefix, i)] = mean_score
    result["{}:pred{}:max_sens".format(prefix, i)] = max_score
    rows_means.append(mean_score)
    rows_maxs.append(max_score)

  # Compute mean and max values across rows.
  column_means = []
  column_maxs = []
  for j in range(metric.shape[1]):
    relevant_scores = [metric[i, j] for i in range(metric.shape[0]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:sens{}:mean_pred".format(prefix, j)] = mean_score
    result["{}:sens{}:max_pred".format(prefix, j)] = max_score
    column_means.append(mean_score)
    column_maxs.append(max_score)

  # Compute all combinations of scores.
  result["{}:mean_sens:mean_pred".format(prefix)] = np.mean(column_means)
  result["{}:mean_sens:max_pred".format(prefix)] = np.mean(column_maxs)
  result["{}:max_sens:mean_pred".format(prefix)] = np.amax(column_means)
  result["{}:max_sens:max_pred".format(prefix)] = np.amax(column_maxs)
  result["{}:mean_pred:mean_sens".format(prefix)] = np.mean(rows_means)
  result["{}:mean_pred:max_sens".format(prefix)] = np.mean(rows_maxs)
  result["{}:max_pred:mean_sens".format(prefix)] = np.amax(rows_means)
  result["{}:max_pred:max_sens".format(prefix)] = np.amax(rows_maxs)

  return result


def inter_group_fairness(counts):
  """Computes the inter group fairness for predictions based on the TV distance.

  Args:
   counts: Numpy array with counts of predictions where rows correspond to
     predicted classes and columns to sensitive classes.

  Returns:
    Mean and maximum total variation distance of a sensitive class to the
      global average.
  """
  # Compute the distribution of predictions across all sensitive classes.
  overall_distribution = np.sum(counts, axis=1, dtype=np.float32)
  overall_distribution /= overall_distribution.sum()

  # Compute the distribution for each sensitive class.
  normalized_counts = np.array(counts, dtype=np.float32)
  counts_per_class = np.sum(counts, axis=0)
  normalized_counts /= np.expand_dims(counts_per_class, 0)

  # Compute the differences and sum up for each sensitive class.
  differences = normalized_counts - np.expand_dims(overall_distribution, 1)
  total_variation_distances = np.sum(np.abs(differences), 0) / 2.

  mean = (total_variation_distances * counts_per_class)
  mean /= counts_per_class.sum()

  return np.sum(mean), np.amax(total_variation_distances)

def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = \
        ground_truth_data.sample(num_points_iter, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = \
        ground_truth_data.sample(num_points_iter, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.

  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


def make_discretizer(target, num_bins=None,
                     discretizer_fn=None):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)



def _histogram_discretize(target, num_bins=None):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def make_predictor_fn(predictor_fn=None):
  """Wrapper that creates classifiers."""
  return predictor_fn


def logistic_regression_cv():
  """Logistic regression with 5 folds cross validation."""
  return linear_model.LogisticRegressionCV(Cs=10,
                                           cv=model_selection.KFold(n_splits=5))


def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return ensemble.GradientBoostingClassifier()

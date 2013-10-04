% an implementation of the coinciding walk kernel (CWK) described in:
%
%   Neumann, M., Garnett, R., and Kersting, K. Coinciding Walk
%   Kernels: Parallel Absorbing Random Walks for Learning with Graphs
%   and Few Labels. (2013). To appear in: Proceedings of the 5th
%   Annual Asian Conference on Machine Learning (ACML 2013).
%
% function [K_train, K_test] = coinciding_walk_kernel(A, labels, ...
%          train_ind, test_ind, walk_length, varargin)
%
% required inputs:
%                 A: the adjacency matrix for the graph under
%                    consideration
%         train_ind: a list of indices into A comprising the
%                    training nodes
%   observed_labels: a list of integer labels corresponding to the
%                    nodes in train_ind
%          test_ind: a list of indices into A comprising the test
%                    nodes
%       walk_length: the maximum walk length for the CWK
%
% optional named arguments specified after requried inputs:
%          'alpha': the absorbtion parameter to use in [0, 1]
%                   (default: 1)
%   'walk_lengths': the set of walk lengths to report the CWK for
%                   (default: {walk_length}).  this can save
%                   computation if multiple walk lengths are to be
%                   compared.
%      'use_prior': a boolean indicating whether to use the
%                   empirical distribution on the training points
%                   as the prior (true) or a uniform prior (false)
%                   (default: false)
%    'pseudocount': if use_prior is set to true, a per-class
%                   pseudocount can also be specified (default: 1)
%
% outputs:
%   K_train: the set of (train x train) kernel matrices.
%            K_train(:, :, i) corresponds to the ith largest
%            specified walk length.
%    K_test: the set of (train x test) kernel matrices.
%            K_test(:, :, i) corresponds to the ith largest
%            specified walk length.
%
% copyright (c) Roman Garnett, 2013.

function [K_train, K_test] = coinciding_walk_kernel(A, train_ind, ...
          observed_labels, test_ind, num_classes, walk_length, varargin)

  options = inputParser;
  options.addParamValue('alpha', 1, ...
                        @(x) (isscalar(x) && (x >= 0) && (x <= 1)));
  options.addParamValue('walk_lengths', walk_length, ...
                        @(x) (isnumeric(x) && all(x >= 0)));
  options.addParamValue('use_prior', false, ...
                        @(x) (islogical(x) && (numel(x) == 1)));
  options.addParamValue('pseudocount', 1, ...
                        @(x) (isscalar(x) && (x > 0)));

  options.parse(varargin{:});
  options = options.Results;

  options.walk_lengths = sort(options.walk_lengths);
  num_walk_lengths = numel(options.walk_lengths);

  num_nodes = size(A, 1);
  num_train = numel(train_ind);
  num_test  = numel(test_ind);

  problem.num_classes = num_classes;
  probabilities = label_propagation_probability(problem, train_ind, obseved_labels, ...
          (1:num_nodes)', A, ...
          'num_iterations',     walk_length, ...
          'alpha',              options.alpha, ...
          'use_prior',          options.use_prior, ...
          'pseudocount',        options.pseudocount, ...
          'store_intermediate', true);

  current_K_train = zeros(num_train, num_train);
  current_K_test  = zeros(num_test,  num_train);

  K_train = zeros(num_train, num_train, num_walk_lengths);
  K_test  = zeros(num_test,  num_train, num_walk_lengths);

  walk_length_ind = 1;
  for i = 1:(walk_length + 1)
    current_K_train = current_K_train + ...
        probabilities(train_ind, :, i) * probabilities(train_ind, :, i)';
    current_K_test  = current_K_test  + ...
        probabilities(test_ind, :, i)  * probabilities(train_ind, :, i)';

    if (ismembc(i - 1, options.walk_lengths))
      K_train(:, :, walk_length_ind) = current_K_train / i;
      K_test( :, :, walk_length_ind) = current_K_test  / i;
      walk_length_ind = walk_length_ind + 1;
    end
  end
end
% MAIN  Semi-supervised minimax risk classifiers (SMRCs) on the USPS dataset.
%
% Runs SMRCs with the same settings as those used for Table 1 in the paper
% (l = 500 labeled samples, u = 7000 unlabeled samples, 1000 test samples,
% K1 = K2 = 50, K3 = K4 = 200) for one random partition of the data, and
% prints the error of the deterministic SMRC, the error of the supervised
% classifier h_sup, and the minimax risk R(V) obtained at learning.
%
% Requires the Statistics and Machine Learning Toolbox (R2021a or later).

clear
close all

rng(1)  % change the seed to obtain other partitions

% Load data (x: d x n matrix with one instance per column, y: labels)
code_dir = fileparts(mfilename('fullpath'));
load(fullfile(code_dir, '..', 'data', 'usps.mat'), 'x', 'y')

x = zscore(x')';
Y = length(unique(y));

n_sup = 500;    % labeled samples
n_uns = 7000;   % unlabeled samples
n_test = 1000;  % test samples
K1 = 50;        % principal components
K2 = 50;        % k-means clusters
K3 = 200;       % hidden neurons
K4 = 200;       % boosted trees

% Random stratified partition of the samples
c = cvpartition(y, 'HoldOut', n_sup);
idx_sup = find(test(c) == 1)';
idx_rest = find(training(c) == 1)';

c2 = cvpartition(y(idx_rest), 'HoldOut', n_uns);
idx_uns = idx_rest(test(c2) == 1);
idx_rest2 = idx_rest(training(c2) == 1);

c3 = cvpartition(y(idx_rest2), 'HoldOut', n_test);
idx_test = idx_rest2(test(c3) == 1);

n_sup = length(idx_sup);
n_uns = length(idx_uns);
n_test = length(idx_test);

% Supervised and unsupervised learning algorithms (lines 1-4 of Algorithm 2)
param_uns = learn_uns(x(:, idx_uns), K1, K2);
param_sup = learn_sup(x(:, idx_sup), y(idx_sup), K3, K4);

% Feature mapping and mean/confidence vectors (lines 5-8 of Algorithm 2)
phi_u = compute_phi_uns(x(:, [idx_sup, idx_uns]), param_uns, Y);
m_u = (K1 + K2 + 1) * Y;

feat = zeros(m_u, n_sup);
for i = 1:n_sup
    feat(:, i) = phi_u{i}(y(idx_sup(i)), :)';
end
tau_u = mean(feat, 2);
lambda_u = std(feat, [], 2) / sqrt(n_sup);

[phi_s, pred] = compute_phi_sup(x(:, [idx_sup, idx_uns]), param_sup, Y);
m_s = K3 * Y;

feat = zeros(m_s, n_uns);
for i = 1:n_uns
    feat(:, i) = phi_s{n_sup + i}(pred(n_sup + i), :)';
end
tau_s = mean(feat, 2);
lambda_s = std(feat, [], 2) / sqrt(n_uns);

tau = [tau_u; tau_s];
lambda = [lambda_u; lambda_s];
phi_semi = cell(1, n_sup + n_uns);
for i = 1:n_sup + n_uns
    phi_semi{i} = horzcat(phi_u{i}, phi_s{i});
end

% Increase confidence vector lambda to ensure the uncertainty set is not
% empty (line 9 of Algorithm 2)
labels = [reshape(double(y(idx_sup)), 1, []), reshape(pred(n_sup + 1:end), 1, [])];
tau_rec = zeros(1, length(tau));
for i = 1:n_sup + n_uns
    tau_rec = tau_rec + phi_semi{i}(labels(i), :) / (n_sup + n_uns);
end
lambda = max(lambda, abs(tau - tau_rec'));

% SMRC parameters and minimax risk (line 10 of Algorithm 2)
iter = 10^7;
B = 1;
[muu, minimax] = fit_SMRC(tau, lambda, phi_semi, Y, iter, B);

% Estimate errors using the test samples
err_SMRC = 0;
err_sup = 0;
for i = 1:n_test
    aux = compute_phi_uns(x(:, idx_test(i)), param_uns, Y);
    aux2 = compute_phi_sup(x(:, idx_test(i)), param_sup, Y);
    phi_x = horzcat(aux{1}, aux2{1});
    [~, pred_det_SMRC] = predict_SMRC(phi_x, muu, Y);
    if pred_det_SMRC ~= y(idx_test(i))
        err_SMRC = err_SMRC + 1 / n_test;
    end
    pred_sup = predict(param_sup.model, x(:, idx_test(i))');
    if pred_sup ~= y(idx_test(i))
        err_sup = err_sup + 1 / n_test;
    end
end

fprintf('SMRC error:            %.3f\n', err_SMRC)
fprintf('Supervised error:      %.3f\n', err_sup)
fprintf('Minimax risk R(V):     %.3f\n', minimax)

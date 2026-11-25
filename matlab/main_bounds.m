% MAIN_BOUNDS  SMRC performance guarantees on the USPS dataset.
%
% Learns an SMRC as in main.m and computes the high-confidence upper and
% lower bounds for its error probability given by Theorem 6 in the paper,
% as done for Figure 1 (l = 2000 labeled samples, delta = 0.05). The bounds
% use the feature mapping Phi_uns, with confidence intervals for its
% expectation obtained by betting (confidence_interval.m).
%
% Note: each stochastic subgradient optimization uses 10^7 iterations, so
% the script can take a long time; reduce `iter` for a quick test.

clear
close all

rng(1)

code_dir = fileparts(mfilename('fullpath'));
load(fullfile(code_dir, '..', 'data', 'usps.mat'), 'x', 'y')

x = zscore(x')';
Y = length(unique(y));

n_sup = 2000;
n_uns = 6000;
n_test = 1000;
K1 = 50;
K2 = 50;
K3 = 200;
K4 = 200;
delta = 0.05;
iter = 10^7;

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

% SMRC learning (as in main.m)
param_uns = learn_uns(x(:, idx_uns), K1, K2);
param_sup = learn_sup(x(:, idx_sup), y(idx_sup), K3, K4);

phi_u = compute_phi_uns(x(:, [idx_sup, idx_uns]), param_uns, Y);
m_u = (K1 + K2 + 1) * Y;
feat_u = zeros(m_u, n_sup);
for i = 1:n_sup
    feat_u(:, i) = phi_u{i}(y(idx_sup(i)), :)';
end
tau_u = mean(feat_u, 2);
lambda_u = std(feat_u, [], 2) / sqrt(n_sup);

[phi_s, pred] = compute_phi_sup(x(:, [idx_sup, idx_uns]), param_sup, Y);
m_s = K3 * Y;
feat_s = zeros(m_s, n_uns);
for i = 1:n_uns
    feat_s(:, i) = phi_s{n_sup + i}(pred(n_sup + i), :)';
end
tau_s = mean(feat_s, 2);
lambda_s = std(feat_s, [], 2) / sqrt(n_uns);

tau = [tau_u; tau_s];
lambda = [lambda_u; lambda_s];
phi_semi = cell(1, n_sup + n_uns);
for i = 1:n_sup + n_uns
    phi_semi{i} = horzcat(phi_u{i}, phi_s{i});
end

labels = [reshape(double(y(idx_sup)), 1, []), reshape(pred(n_sup + 1:end), 1, [])];
tau_rec = zeros(1, length(tau));
for i = 1:n_sup + n_uns
    tau_rec = tau_rec + phi_semi{i}(labels(i), :) / (n_sup + n_uns);
end
lambda = max(lambda, abs(tau - tau_rec'));

[muu, minimax] = fit_SMRC(tau, lambda, phi_semi, Y, iter, 1);

% Mean and confidence vectors for Phi_uns with confidence 1 - delta
% (union bound over the m_u components)
tau_conf = zeros(m_u, 1);
lambda_conf = zeros(m_u, 1);
for i = 1:m_u
    a = min(feat_u(i, :));
    b = max(feat_u(i, :));
    if b > a
        [t, l] = confidence_interval((feat_u(i, :) - a) / (b - a), delta / m_u, 100);
        tau_conf(i) = (b - a) * t + a;
        lambda_conf(i) = (b - a) * l;
    else
        tau_conf(i) = a;
        lambda_conf(i) = 0;
    end
end
% Ensure the optimization problems in Theorem 6 are bounded
tau_rec_u = tau_rec(1:m_u)';
lambda_conf = max(lambda_conf, abs(tau_conf - tau_rec_u));

% Deterministic SMRC evaluated at the training instances
h_x = zeros(Y, n_sup + n_uns);
for i = 1:n_sup + n_uns
    [~, pred_det] = predict_SMRC(phi_semi{i}, muu, Y);
    h_x(pred_det, i) = 1;
end
[lower, upper] = performance_bounds(tau_conf, lambda_conf, phi_u, h_x, iter);

% Test error
err_SMRC = 0;
for i = 1:n_test
    aux = compute_phi_uns(x(:, idx_test(i)), param_uns, Y);
    aux2 = compute_phi_sup(x(:, idx_test(i)), param_sup, Y);
    [~, pred_det] = predict_SMRC(horzcat(aux{1}, aux2{1}), muu, Y);
    if pred_det ~= y(idx_test(i))
        err_SMRC = err_SMRC + 1 / n_test;
    end
end

fprintf('SMRC error:                    %.3f\n', err_SMRC)
fprintf('Minimax risk R(V):             %.3f\n', minimax)
fprintf('High-confidence upper bound:   %.3f\n', upper)
fprintf('High-confidence lower bound:   %.3f\n', lower)

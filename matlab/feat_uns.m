function x_feat_uns = feat_uns(x, param)
% FEAT_UNS  Features Psi_uns(x) learned by learn_uns.m or
% learn_uns_autoencoder.m.
%
%   x_feat_uns = FEAT_UNS(x, param)
%
%   x          : d x n matrix of instances
%   x_feat_uns : D1 x n matrix of features [1; projections; clusters]

n = size(x, 2);
K2 = size(param.C, 1);

% One-hot encoding of the nearest centroid
xx = zeros(K2, n);
for i = 1:n
    dist = vecnorm(param.C - x(:, i)', 2, 2);
    [~, idx] = min(dist);
    xx(idx, i) = 1;
end

if isfield(param, 'autoenc')
    xxx = encode(param.autoenc, x);
else
    xxx = param.coeff * (x - repmat(param.mu, 1, n));
end

x_feat_uns = [ones(1, n); xxx; xx];
end

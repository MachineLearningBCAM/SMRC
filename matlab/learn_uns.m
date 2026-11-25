function param = learn_uns(x_uns, K1, K2)
% LEARN_UNS  Unsupervised feature learning with PCA and k-means clustering.
%
%   param = LEARN_UNS(x_uns, K1, K2)
%
%   x_uns : d x u matrix of unlabeled instances
%   K1    : number of principal components
%   K2    : number of k-means clusters
%
%   The features Psi_uns(x) (see feat_uns.m) are given by a constant, the
%   projections onto the first K1 principal components (scaled to unit
%   variance), and the one-hot encoding of the nearest of K2 centroids.

[coeff, score, ~, ~, ~, mu] = pca(x_uns');

param.coeff = zeros(K1, size(x_uns, 1));
for i = 1:K1
    param.coeff(i, :) = (coeff(:, i)') / std(score(:, i));
end
param.mu = mu';

[~, param.C] = kmeans(x_uns', K2);
end

function param = learn_uns_autoencoder(x_uns, K5, K2)
% LEARN_UNS_AUTOENCODER  Unsupervised feature learning with an autoencoder
% and k-means clustering (setting used for Figure 1(b) in the paper).
%
%   param = LEARN_UNS_AUTOENCODER(x_uns, K5, K2)
%
%   x_uns : d x u matrix of unlabeled instances
%   K5    : size of the autoencoder's hidden layer
%   K2    : number of k-means clusters
%
%   The features Psi_uns(x) (see feat_uns.m) are given by a constant, the
%   K5 outputs of the hidden layer of an autoencoder with logistic sigmoid
%   transfer function, and the one-hot encoding of the nearest of K2
%   centroids. Requires the Deep Learning Toolbox.

param.autoenc = trainAutoencoder(x_uns, K5, 'ShowProgressWindow', false);
[~, param.C] = kmeans(x_uns', K2);
end

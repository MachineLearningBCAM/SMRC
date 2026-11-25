function x_feat_sup = feat_sup(x, param)
% FEAT_SUP  Features Psi_sup(x) learned by learn_sup.m: standardized
% outputs of the first layer of the neural network.
%
%   x          : d x n matrix of instances
%   x_feat_sup : K3 x n matrix of features

n = size(x, 2);
x_feat_sup = param.D * (param.W * x - repmat(param.mu, 1, n));
end

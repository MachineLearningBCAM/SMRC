function phi = compute_phi_uns(x, param, Y)
% COMPUTE_PHI_UNS  Feature mapping Phi_uns(x, y) = e_y (x) Psi_uns(x).
%
%   phi = COMPUTE_PHI_UNS(x, param, Y)
%
%   x     : d x n matrix of instances
%   param : unsupervised algorithm learned by learn_uns.m
%   Y     : number of classes
%
%   phi   : cell array; phi{i} is the Y x m1 matrix with rows
%           Phi_uns(x_i, 1), ..., Phi_uns(x_i, Y)

n = size(x, 2);
I = eye(Y);
phi = cell(1, n);

for i = 1:n
    xx = feat_uns(x(:, i), param);
    N = zeros(Y, Y * length(xx));
    for k = 1:Y
        N(k, :) = kron(I(:, k), xx)';
    end
    phi{i} = N;
end
end

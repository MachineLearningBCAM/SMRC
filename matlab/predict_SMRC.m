function [y, y_det, h] = predict_SMRC(phi_x, muu, Y)
% PREDICT_SMRC  Evaluate an SMRC at an instance x.
%
%   [y, y_det, h] = PREDICT_SMRC(phi_x, muu, Y)
%
%   phi_x : Y x m matrix with rows Phi(x, 1), ..., Phi(x, Y)
%   muu   : SMRC parameters (m x 1)
%   Y     : number of classes
%
%   y     : label sampled from the randomized SMRC h(.|x), eq. (3.3)
%   y_det : label assigned by the deterministic SMRC, eq. (4.4)
%   h     : probabilities h(y|x) = (Phi(x, y)' * muu - varphi(muu, x))_+

[~, varphi] = subgradient(phi_x, muu, Y);
v = phi_x * muu;
h = max(v - varphi, 0);

y = find(mnrnd(1, h / sum(h)) == 1);
[~, y_det] = max(h);
end

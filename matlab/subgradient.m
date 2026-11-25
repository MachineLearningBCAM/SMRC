function [grad, varphi, psi] = subgradient(phi_x, muu, Y)
% SUBGRADIENT  Algorithm 1 in the paper (efficient computation of subgradients).
%
%   [grad, varphi, psi] = SUBGRADIENT(phi_x, muu, Y)
%
%   phi_x  : Y x m matrix with rows Phi(x, 1), ..., Phi(x, Y)
%   muu    : parameters (m x 1)
%   Y      : number of classes
%
%   grad   : subgradient of varphi(., x) at muu (m x 1)
%   varphi : varphi(muu, x) = max_C (sum_{y in C} Phi(x, y)' * muu - 1) / |C|
%   psi    : psi(muu, x) = varphi(muu, x) - (max_y Phi(x, y)' * muu - 1),
%            used in the bound for deterministic SMRCs (Theorem 4)

v = phi_x * muu;

[~, ord] = sort(v, 'descend');

varphi = v(ord(1)) - 1;
psi = 0;
grad = phi_x(ord(1), :)';

for k = 2:Y
    value_new = ((k - 1) * varphi + v(ord(k))) / k;
    if value_new >= varphi
        varphi = value_new;
        psi = varphi - (v(ord(1)) - 1);
        grad = ((k - 1) * grad + phi_x(ord(k), :)') / k;
    else
        break
    end
end
end

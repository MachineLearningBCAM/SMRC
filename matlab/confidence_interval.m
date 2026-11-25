function [tau, lambda] = confidence_interval(x, alpha, k)
% CONFIDENCE_INTERVAL  Confidence interval for the mean of samples in [0, 1]
% using the betting approach in Waudby-Smith and Ramdas, "Estimating means
% of bounded random variables by betting", JRSS-B, 2024.
%
%   [tau, lambda] = CONFIDENCE_INTERVAL(x, alpha, k)
%
%   x      : vector of samples in [0, 1]
%   alpha  : the interval contains the mean with probability >= 1 - alpha
%   k      : number of grid points used for the candidate means
%
%   The interval is [tau - lambda, tau + lambda].

n = length(x);
mu = zeros(1, n);
v = zeros(1, n);
lambda_dot = zeros(1, n);
for i = 1:n
    mu(i) = (0.5 + sum(x(1:i))) / (i + 1);
    v(i) = ((1/4) + sum((x(1:i) - mu(1:i)).^2)) / (i + 1);
    if i == 1
        lambda_dot(i) = sqrt((2 * log(2 / alpha)) / (n * (1/4)));
    else
        lambda_dot(i) = sqrt((2 * log(2 / alpha)) / (n * v(i - 1)));
    end
end

grid = linspace(0, 1, k);
K = zeros(k, n);
for i = 1:k
    lambda_1 = min(abs(lambda_dot), 0.5 / grid(i));
    lambda_2 = min(abs(lambda_dot), 0.5 / (1 - grid(i)));
    K_1 = cumprod(1 + lambda_1 .* (x(:)' - grid(i)));
    K_2 = cumprod(1 - lambda_2 .* (x(:)' - grid(i)));
    K(i, :) = max(0.5 * K_1, 0.5 * K_2);
end

b = k;
a = 1;
for i = 1:n
    idx = find(K(:, i) < 1 / alpha);
    if ~isempty(idx)
        b = min(b, max(idx));
        a = max(a, min(idx));
    end
end
tau = (grid(b) + grid(a)) / 2;
lambda = max(grid(b) - tau, grid(a) - tau);
end

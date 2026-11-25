function param = learn_sup(x_sup, y_sup, K3, K4)
% LEARN_SUP  Supervised feature learning and classification.
%
%   param = LEARN_SUP(x_sup, y_sup, K3, K4)
%
%   x_sup : d x l matrix of labeled instances
%   y_sup : labels in {1, ..., Y}
%   K3    : number of hidden neurons of the feedforward neural network that
%           provides the features Psi_sup(x) (see feat_sup.m)
%   K4    : number of trees in the boosted ensemble that provides the
%           classifier h_sup
%
%   The set of confidently classified instances is given by the instances
%   classified with a score larger than param.th, which is the percentile of
%   the scores corresponding to the error estimated by 10-fold
%   cross-validation (Section 5.2 in the paper).

% Supervised classifier h_sup and confidence threshold
param.model = fitcensemble(x_sup', y_sup, 'ScoreTransform', 'doublelogit', ...
    'NumLearningCycles', K4);
model_aux = fitcensemble(x_sup', y_sup, 'ScoreTransform', 'doublelogit', ...
    'CrossVal', 'on', 'KFold', 10, 'NumLearningCycles', K4);

% For simplicity, the confidence scores are obtained with the 'doublelogit'
% transform. Note that such scores may saturate at 1 for boosted ensembles,
% in which case all instances are taken as confidently classified.
err = kfoldLoss(model_aux);
[pred, scores] = kfoldPredict(model_aux);
score = zeros(1, length(pred));
for i = 1:length(pred)
    score(i) = scores(i, pred(i));
end
param.th = prctile(score, err * 100) - 10^-4;

% Supervised features: first layer of a feedforward neural network
mod = fitcnet(x_sup', y_sup, 'LayerSizes', K3);

W = mod.LayerWeights{1};
F = W * x_sup;
param.W = W;
param.mu = mean(F, 2);
param.D = diag(1 ./ std(F, [], 2));
end

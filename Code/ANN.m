clear all;
rng(0); % random seed setting

% Data pre-processing to scale data into 0-1 and cleaning
% porosity is without any treatment
% tortuosity = 1/tortuosity_raw
% permeability = log10(permeability_raw)/(-15)
% Any data >1 or <0 is cleaned

% load dataset
data = readtable('folder_path\shuffled_scaled_dataset.csv');

% feature and target selection 
features = data{:, 1:3}; % input
targets = data{:, 4:end}; % output

% divide dataset
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;
cv = cvpartition(size(features, 1), 'HoldOut', valRatio + testRatio);
trainIdx = training(cv);
tempIdx = test(cv);

cvValTest = cvpartition(sum(tempIdx), 'HoldOut', testRatio / (valRatio + testRatio));
valIdx = false(size(features, 1), 1);
valIdx(tempIdx) = training(cvValTest);

testIdx = false(size(features, 1), 1);
testIdx(tempIdx) = test(cvValTest);

XTrain = features(trainIdx, :);
YTrain = targets(trainIdx, :);
XValidation = features(valIdx, :);
YValidation = targets(valIdx, :);
XTest = features(testIdx, :);
YTest = targets(testIdx, :);

% building Neural Network
layers = [ ...
    featureInputLayer(3)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(size(targets, 2))
    regressionLayer];

% trainingOptions
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation, YValidation}, ...
    'Plots', 'training-progress');

% Training
[net, info ]=trainNetwork(XTrain, YTrain, layers, options);


% Prediction performance
YPred = predict(net, XTest);

% Calculate R2 score on Test
r2 = @(y, f) 1 - sum((y - f).^2) / sum((y - mean(y)).^2);
R2_score = r2(YTest, YPred);

% plot
figure;
plot(YTest, YPred, 'bo');
hold on;
plot([min(YTest) max(YTest)], [min(YTest) max(YTest)], 'k--');
xlabel('Ground Truth');
ylabel('Predictions');
title(sprintf('Test Set Predictions (R^2 = %.2f)', R2_score));
grid on;

% save NN
% save('folder_path\saved_net.mat', 'net');


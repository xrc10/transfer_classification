clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

% folder_name = 'TDT5_English_svm';
folder_name = 'TDT5_Chinese_svm_withDict';

[yTrn, XTrn] = libsvmread(['../data/', folder_name, '/trn.svm']);
[yTst, XTst] = libsvmread(['../data/', folder_name, '/tst.svm']);

labels = unique(yTrn);

%% evaluation on test set directly
macroF1 = 0;
microF1 = 0;
repTime = 40;
for i = 1:repTime
    randIdx = randi([1;size(labels,1)], size(yTst,1), 1);
    yPred = labels(randIdx);
    evalObj = evaluate(yTst, yPred);
    macroF1 = macroF1 + evalObj.macroF1;
    microF1 = microF1 + evalObj.microF1;
end
fprintf('macro F1 is %f, micro F1 is %f\n', macroF1/repTime, microF1/repTime);

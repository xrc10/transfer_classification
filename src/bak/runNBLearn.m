clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

folder_name = 'TDT5_Chinese_wordcount_withDict';
% folder_name = 'TDT5_English_wordcount';
% folder_name = 'TDT5_English_wordcount_withDict';

[yTrn, XTrn] = libsvmread(['../data/', folder_name, '/trn.svm']);
[yVal, XVal] = libsvmread(['../data/', folder_name, '/val.svm']);
[yTst, XTst] = libsvmread(['../data/', folder_name, '/tst.svm']);

p = max([size(XTrn,2), size(XVal,2), size(XTst,2)]);

XTrn = extSparseDim(XTrn, 2, p);
XVal = extSparseDim(XVal, 2, p);
XTst = extSparseDim(XTst, 2, p);

%% train on train set and tune C on validation set 
fprintf('Training with Naive Bayes...,');
mod = fitNaiveBayes(XTrn, yTrn, 'Distribution', 'mn');
yPred = predict(mod, XVal);
evalObj = evaluate(yVal, yPred);
fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);

%% train with best hyperparameter
yPred = predict(mod, XTst);
evalObj = evaluate(yTst, yPred);
fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
save(['../data/', folder_name, '/model'], 'mod');


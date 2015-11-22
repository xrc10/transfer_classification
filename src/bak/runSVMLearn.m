clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

% folder_name = 'TDT5_Chinese_svm';
folder_name = 'TDT5_Chinese_svm_withDict';
% folder_name = 'TDT5_English_svm';
% folder_name = 'TDT5_English_svm_withDict';

[yTrn, XTrn] = libsvmread(['../data/', folder_name, '/trn.svm']);
[yVal, XVal] = libsvmread(['../data/', folder_name, '/val.svm']);
[yTst, XTst] = libsvmread(['../data/', folder_name, '/tst.svm']);

p = max([size(XTrn,2), size(XVal,2), size(XTst,2)]);

XTrn = extSparseDim(XTrn, 2, p);
XVal = extSparseDim(XVal, 2, p);
XTst = extSparseDim(XTst, 2, p);

%% train on train set and tune C on validation set 
cList = 10.^(-3:3);
addpath('../../liblinear/');
microF1List = zeros(size(cList,2), 1);
for c = cList
    fprintf('tunning on C:%f...,', c);
    mod = train(yTrn, XTrn, ['-s 4 -c ',num2str(c), ' -q']);
    yPred = predict(yVal, XVal, mod, '-q');
    evalObj = evaluate(yVal, yPred);
    fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    microF1List(cList == c) = evalObj.microF1;
end

%% find the best hyperparameter
[bestF1, bestIdx] = max(microF1List(:));
fprintf('best C is %f\n', cList(bestIdx));

%% train with best hyperparameter
mod = train(yTrn, XTrn, ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
yPred = predict(yTst, XTst, mod, '-q');
evalObj = evaluate(yTst, yPred);
fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
save(['../data/', folder_name, '/model'], 'mod');


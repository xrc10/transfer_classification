clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

p = 62061;

[yTrn, XTrn] = libsvmread('../data/20news/news20.svm');
XTrn = extSparseDim(XTrn, 2, p);
[yTst, XTst] = libsvmread('../data/20news/news20.t.svm');
XTst = extSparseDim(XTst, 2, p);
%% train on train set and tune C on validation set 
cList = 10.^(-4:1:8);
addpath('../../liblinear/');

for c = cList
    fprintf('tunning on C:%f...\n', c);
    mod = train(yTrn, XTrn, ['-s 0 -c ',num2str(c), ' -q']);
    yPred = predict(yTst, XTst, mod, '-q');
    evalObj = evaluate(yTst, yPred);
    fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
end




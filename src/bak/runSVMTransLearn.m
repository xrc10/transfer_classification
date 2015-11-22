clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

pSrc = 17327;
pTgt = 74539;

[yTst, XTst] = libsvmread('../data/TDT5_Chinese_svm_withDict/tst.svm');
[yTrn, ~] = libsvmread('../data/TDT5_Chinese_svm_withDict/trn.svm');

XTst = extSparseDim( XTst, 2, pTgt );

labels = unique(yTrn);

load('../data/TDT5_English_svm_withDict/model.mat');
wEng = (mod.w)';
wEng = [wEng; zeros(pSrc - size(wEng,1), size(wEng,2))];
% labelAssign = [1;2;3;0;4;0;0;7;0;8];
% activeLabels = labels(labelAssign ~= 0);
% activeInst = ismember(yTst, activeLabels);
% XTst = XTst(activeInst,:);
% yTst = yTst(activeInst);

%% transfer 
% load('../data/linear_WE_transfer/simDictM.mat', 'simM');
% load('../data/linear_WE_transfer/cosSimM.mat', 'simM');
load('../data/linear_WE_transfer/eucSimM.mat', 'simM');

% normalize simM
simM = simM';
n =  sum( simM, 2 ) ;
n( n == 0 ) = 1;
fprintf('normalizing similarity matrix...\n');
simM = bsxfun( @rdivide, simM, n );
%% simple weighted summation
fprintf('transfering SVM model...\n');
wChn = simM * wEng;

%% evaluation on test set directly
[~, maxIdx] = max(XTst * wChn, [], 2);
yPred = labels(maxIdx);
evalObj = evaluate(yTst, yPred);
fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
save('eval.mat', 'evalObj');
save('wChn.mat', 'wChn');


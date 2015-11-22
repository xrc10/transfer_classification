clc; clear all; close all;

%% read data

fprintf('loading data...\n');

addpath('../tool');

pSrc = 17327;
pTgt = 74539;

[yTst, XTst] = libsvmread('../data/TDT5_Chinese_wordcount_withDict/tst.svm');
[yTrn, ~] = libsvmread('../data/TDT5_Chinese_wordcount_withDict/trn.svm');

XTst = extSparseDim(XTst, 2, pTgt);

labels = unique(yTrn);

load('../data/TDT5_English_wordcount_withDict/model.mat');
modTgt.NClasses = mod.NClasses;
modTgt.NDims = pTgt;
modTgt.ClassLevels = labels;
modTgt.CIsNonEmpty = mod.CIsNonEmpty;
modTgt.Dist = mod.Dist;
modTgt.Prior = mod.Prior;
modTgt.NonEmptyClasses = (1:size(labels,1))';
paramsSrc = extSparseDim(cell2mat(mod.Params), 2, pSrc);

%% transfer 
% load('../data/linear_WE_transfer/simDictM.mat', 'simM');
% load('../data/linear_WE_transfer/cosSimM.mat', 'simM');
load('../data/linear_WE_transfer/eucSimM.mat', 'simM');

% normalize simM
n =  sum( simM, 2 );
n( n == 0 ) = 1;
fprintf('normalizing similarity matrix...\n');
simM = bsxfun( @rdivide, simM, n );
%% simple weighted summation
fprintf('transfering NB model...\n');
paramsTgt = paramsSrc * simM;
paramsTgt = paramsTgt + 1/size(paramsTgt, 2);
%% normalize paramsTgt
n =  sum( paramsTgt, 2 ) ;
n( n == 0 ) = 1;
fprintf('normalizing parameter matrix...\n');
paramsTgt = bsxfun( @rdivide, paramsTgt, n );

modTgt.Params = {paramsTgt};

%% evaluation on test set directly
yPred = myNBPredict(modTgt, XTst);
evalObj = evaluate(yTst, yPred);
fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
mod = modTgt;
save('../data/TDT5_Chinese_wordcount_withDict/modTgt.mat', 'mod');


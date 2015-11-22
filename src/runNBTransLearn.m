clc; clear all; close all;

%% read data

tgtFolder = 'TDT5_Chinese_event_dict_wordcount';
srcFolder = 'TDT5_English_event_dict_wordcount';
repTime = 100;
pSrc = 17327;
pTgt = 74539;
% simMPath = '../data/linear_WE_transfer/eucSimM.mat';
simMPath = '../data/linear_WE_transfer/simDictM.mat';
% simMPath = '../data/linear_WE_transfer/cosSimM.mat';

if ~exist(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder]), 'dir')
    mkdir(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder]));
end
logFile = fopen(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder], 'log.txt'), 'w');

fprintf('loading data...\n');

addpath('../tool');

classes = {'Accidents',...
    'Acts of Violence or War',...
    'Celebrity and Human Interest News',...
    'Legal and Criminal Cases',...
    'Natural Disasters',...
    'Political and Diplomatic Meetings',...
    'Sports News'};

avgMacroF1 = zeros(repTime, 1);
avgMicroF1 = zeros(repTime, 1);

for i = 1:repTime
    
    dataSplit = randBiSplit(['../data/', srcFolder], ['../data/', tgtFolder], classes);
    
    %% train on src train set
    fprintf('Training with Naive Bayes...,');
    mod = NaiveBayes.fit(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    %     mod = fitNaiveBayes(dataSplit.XTrn, dataSplit.yTrn, 'Distribution', 'mn');
    
    %% transfer the learned model
    labels = unique(dataSplit.srcYTrn);
    modTgt.NClasses = mod.NClasses;
    modTgt.NDims = pTgt;
    modTgt.ClassLevels = labels;
    modTgt.CIsNonEmpty = mod.CIsNonEmpty;
    modTgt.Dist = mod.Dist;
    modTgt.Prior = mod.Prior;
    modTgt.NonEmptyClasses = (1:size(labels,1))';
    paramsSrc = extSparseDim(cell2mat(mod.Params), 2, pSrc);
    load(simMPath, 'simM');
    paramsTgt = transNBModelParams( paramsSrc, simM );
    modTgt.Params = {paramsTgt};
    
    %% test on tgt val set for reference
    yPred = predict(modTgt, dataSplit.tgtXVal);
    evalObj = evaluate(dataSplit.tgtYVal, yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test with tgt test set
    yPred = predict(modTgt, dataSplit.tgtXTst);
    evalObj = evaluate(dataSplit.tgtYTst, yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgMacroF1(i) = evalObj.macroF1;
    avgMicroF1(i) = evalObj.microF1;
end

fprintf('-------Overall--------\n');
fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));

fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% normalize simM
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


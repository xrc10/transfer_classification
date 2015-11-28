clc; clear all; close all;

rng(315);

%% read data

tgtFolder = 'TDT5_Chinese_event_dict_wordcount';
srcFolder = 'TDT5_English_event_dict_wordcount';
% srcFolder = 'TDT5_Chinese_event_dict_wordcount';
% tgtFolder = 'TDT5_English_event_dict_wordcount';
repTime = 1000;
pSrc = 17327;
% pSrc = 74539;
pTgt = 74539;
% pTgt = 17327;
% simMPath = '../data/linear_WE_transfer/eucSimM.mat';
simMPath = '../data/linear_WE_transfer/simDictM.mat';
% simMPath = '../data/linear_WE_transfer/cosSimM.mat';

%% load simM
load(simMPath, 'simM');
% simM = simM';
%% normalize simM that each row sum up to 1
n =  sum( simM, 2 );
n( n == 0 ) = 1;
fprintf('normalizing similarity matrix...\n');
simM = bsxfun( @rdivide, simM, n );

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

avgSrcMacroF1 = zeros(repTime, 1);
avgSrcMicroF1 = zeros(repTime, 1);
avgTgtMacroF1 = zeros(repTime, 1);
avgTgtMicroF1 = zeros(repTime, 1);
avgTransMacroF1 = zeros(repTime, 1);
avgTransMicroF1 = zeros(repTime, 1);

for i = 1:repTime
    
    dataSplit = randBiSplit(['../data/', srcFolder], ['../data/', tgtFolder], classes);
    
    %%%%%%%%%%%%%%%src%%%%%%%%%%%%%%%%
    
    %% train on src train set
    fprintf('Training with Naive Bayes...,');
    model = NaiveBayes.fit(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    %     mod = fitNaiveBayes(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    
    %% test on src val set
    yPred = predict(model, dataSplit.srcXVal);
    evalObj = evaluate(dataSplit.srcYVal, yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test on src tst set
    yPred = predict(model, dataSplit.srcXTst);
    evalObj = evaluate(dataSplit.srcYTst, yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgSrcMacroF1(i) = evalObj.macroF1;
    avgSrcMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%trans%%%%%%%%%%%%%%%%
    
    %% transfer the learned model
    labels = unique(dataSplit.srcYTrn);
    modTgt.NClasses = model.NClasses;
    modTgt.NDims = pTgt;
    modTgt.ClassLevels = labels;
    modTgt.CIsNonEmpty = model.CIsNonEmpty;
    modTgt.Dist = model.Dist;
    modTgt.Prior = model.Prior;
    modTgt.NonEmptyClasses = (1:size(labels,1))';
    paramsSrc = extSparseDim(cell2mat(model.Params), 2, pSrc);

    paramsTgt = transNBModelParams( paramsSrc, simM );
    modTgt.Params = {paramsTgt};
    
    %% test on tgt val set for reference
    tgtXVal = extSparseDim(dataSplit.tgtXVal, 2, pTgt);
    yPred = myNBPredict(modTgt, tgtXVal);
    evalObj = evaluate(dataSplit.tgtYVal, yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test with tgt test set
    tgtXTst = extSparseDim(dataSplit.tgtXTst, 2, pTgt);
    yPred = myNBPredict(modTgt, tgtXTst);
    evalObj = evaluate(dataSplit.tgtYTst, yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgTransMacroF1(i) = evalObj.macroF1;
    avgTransMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%tgt%%%%%%%%%%%%%%%%
    
    %% train on tgt train set
    fprintf('Training with Naive Bayes...,');
    model = NaiveBayes.fit(dataSplit.tgtXTrn, dataSplit.tgtYTrn, 'Distribution', 'mn');
    %     mod = fitNaiveBayes(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    
    %% test on tgt val set
    yPred = predict(model, dataSplit.tgtXVal);
    evalObj = evaluate(dataSplit.tgtYVal, yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test on tgt tst set
    yPred = predict(model, dataSplit.tgtXTst);
    evalObj = evaluate(dataSplit.tgtYTst, yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgTgtMacroF1(i) = evalObj.macroF1;
    avgTgtMicroF1(i) = evalObj.microF1;
end
aggMacroF1 = [avgSrcMacroF1, avgTgtMacroF1, avgTransMacroF1];
aggMicroF1 = [avgSrcMicroF1, avgTgtMicroF1, avgTransMicroF1];
tag = {'src', 'tgt', 'trans'};

for i = 1:3
    
    avgMacroF1 = aggMacroF1(:,i);
    avgMicroF1 = aggMicroF1(:,i);
    
    fprintf(['-------', tag{i}, ' Overall--------\n']);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 1.644853*sqrt(var(avgMacroF1)/repTime), 1.644853*sqrt(var(avgMicroF1)/repTime));
    
    fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 1.644853*sqrt(var(avgMacroF1)/repTime), 1.644853*sqrt(var(avgMicroF1)/repTime));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



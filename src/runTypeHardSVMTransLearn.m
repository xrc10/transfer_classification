clc; clear all; close all;

%% read data

tgtFolder = 'TDT5_Chinese_event_dict_wordcount';
srcFolder = 'TDT5_English_event_dict_wordcount';
repTime = 1000;
pSrc = 17327;
pTgt = 74539;
cList = 10.^(-4:0);
% simMPath = '../data/linear_WE_transfer/eucSimM.mat';
simMPath = '../data/linear_WE_transfer/simDictM.mat';
% simMPath = '../data/linear_WE_transfer/cosSimM.mat';

%% load simM
load(simMPath, 'simM');
simM = simM';
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
addpath('../../liblinear/');

for i = 1:repTime
    
    dataSplit = randBiSplit(['../data/', srcFolder], ['../data/', tgtFolder], classes);
    labels = unique(dataSplit.srcYTrn);
    
    %%%%%%%%%%%%%%%src%%%%%%%%%%%%%%%%
    
    %% train on src train set
    fprintf('Training with SVM...,');
    
    microF1List = zeros(size(cList,2), 1);
    
    for c = cList
        fprintf('tunning on C:%f...,', c);
        model = train(dataSplit.srcYTrn, dataSplit.srcXTrn, ['-s 4 -c ',num2str(c), ' -q']);
        yPred = predict(dataSplit.srcYVal, dataSplit.srcXVal, model, '-q');
        evalObj = evaluate(dataSplit.srcYVal, yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        microF1List(cList == c) = evalObj.microF1;
    end
    
    %% find the best hyperparameter
    [bestF1, bestIdx] = max(microF1List(:));
    fprintf('best micro F1 is %f, best C is %f\n', bestF1, cList(bestIdx));
    
    %% train with best hyperparameter
    model = train(dataSplit.srcYTrn, dataSplit.srcXTrn, ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
    
    %% test on src tst set
    yPred = predict(dataSplit.srcYTst, dataSplit.srcXTst, model, '-q');
    evalObj = evaluate(dataSplit.srcYTst, yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgSrcMacroF1(i) = evalObj.macroF1;
    avgSrcMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%trans%%%%%%%%%%%%%%%%
    
    wSrc = (model.w)';
    wSrc = [wSrc; zeros(pSrc - size(wSrc,1), size(wSrc,2))];
    wTgt = simM * wSrc;
    dataSplit.tgtXTst = extSparseDim(dataSplit.tgtXTst, 2, pTgt);
    [~, maxIdx] = max(dataSplit.tgtXTst * wTgt, [], 2);
    yPred = labels(maxIdx);
    evalObj = evaluate(dataSplit.tgtYTst, yPred);
    fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgTransMacroF1(i) = evalObj.macroF1;
    avgTransMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%tgt%%%%%%%%%%%%%%%%
    %% train on src train set
    fprintf('Training with SVM...,');
    
    microF1List = zeros(size(cList,2), 1);
    
    for c = cList
        fprintf('tunning on C:%f...,', c);
        model = train(dataSplit.tgtYTrn, dataSplit.tgtXTrn, ['-s 4 -c ',num2str(c), ' -q']);
        yPred = predict(dataSplit.tgtYVal, dataSplit.tgtXVal, model, '-q');
        evalObj = evaluate(dataSplit.tgtYVal, yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        microF1List(cList == c) = evalObj.microF1;
    end
    
    %% find the best hyperparameter
    [bestF1, bestIdx] = max(microF1List(:));
    fprintf('best micro F1 is %f, best C is %f\n', bestF1, cList(bestIdx));
    
    %% train with best hyperparameter
    model = train(dataSplit.tgtYTrn, dataSplit.tgtXTrn, ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
    
    %% test on src tst set
    yPred = predict(dataSplit.tgtYTst, dataSplit.tgtXTst, model, '-q');
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


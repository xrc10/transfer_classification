clc; clear all; close all;

%% read data

rng(315);

tgtFolder = 'TDT5_Chinese_event_dict_wordcount';
srcFolder = 'TDT5_English_event_dict_wordcount';
% srcFolder = 'TDT5_Chinese_event_dict_wordcount';
% tgtFolder = 'TDT5_English_event_dict_wordcount';
repTime = 100;
pSrc = 17327;
% pSrc = 74539;
pTgt = 74539;
% pTgt = 17327;
% simMPath = '../data/linear_WE_transfer/eucSimM.mat';
simMPath = '../data/linear_WE_transfer/simDictM.mat';
% simMPath = '../data/linear_WE_transfer/cosSimM.mat';

kFold = 5;
cList = 10.^(-3:0);

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
addpath('../../liblinear/');

classes = {'Accidents',...
    'Acts of Violence or War',...
    'Celebrity and Human Interest News',...
    'Elections',...
    'Financial News',...
    'Legal and Criminal Cases',...
    'Miscellaneous News',...
    'Natural Disasters',...
    'Political and Diplomatic Meetings',...
    'Science and Discovery News',...
    'Sports News'};

avgSrcMacroF1 = zeros(kFold, 1);
avgSrcMicroF1 = zeros(kFold, 1);
avgTgtMacroF1 = zeros(kFold, 1);
avgTgtMicroF1 = zeros(kFold, 1);
avgTransMacroF1 = zeros(kFold, 1);
avgTransMicroF1 = zeros(kFold, 1);

dataSplit = kFoldTypeBiSplit(['../data/', srcFolder], ['../data/', tgtFolder], classes, kFold);

for i = 1:kFold
    
    valFold = i;
    tstFold = mod(i, kFold)+1;
    trnFolds = setdiff((1:kFold)', [valFold; tstFold]);
    
    %%%%%%%%%%%%%%%src%%%%%%%%%%%%%%%%
    microF1List = zeros(size(cList,2), 1);
    %% train on src train set
    for c = cList
        fprintf('tunning on C:%f...,', c);
        model = train(cell2mat(dataSplit.srcY(trnFolds)), cell2mat(dataSplit.srcX(trnFolds)), ['-s 4 -c ',num2str(c), ' -q']);
        yPred = predict(cell2mat(dataSplit.srcY(valFold)), cell2mat(dataSplit.srcX(valFold)), model, '-q');
        evalObj = evaluate(cell2mat(dataSplit.srcY(valFold)), yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        microF1List(cList == c) = evalObj.microF1;
    end

    %% find the best hyperparameter
    [bestF1, bestIdx] = max(microF1List(:));
    fprintf('best micro F1 is %f, best C is %f\n', bestF1, cList(bestIdx));
    
    %% test on src tst set
    model = train(cell2mat(dataSplit.srcY(trnFolds)), cell2mat(dataSplit.srcX(trnFolds)), ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
    yPred = predict(cell2mat(dataSplit.srcY(tstFold)), cell2mat(dataSplit.srcX(tstFold)), model, '-q');
    evalObj = evaluate(cell2mat(dataSplit.srcY(tstFold)), yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgSrcMacroF1(i) = evalObj.macroF1;
    avgSrcMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%trans%%%%%%%%%%%%%%%%
    
    %% transfer the learned model
    labels = unique(cell2mat(dataSplit.srcY));
    wSrc = (model.w)';
    wSrc = [wSrc; zeros(pSrc - size(wSrc,1), size(wSrc,2))];
    wTgt = simM * wSrc;
    
    %% test on tgt val set for reference
    tgtXVal = extSparseDim(cell2mat(dataSplit.tgtX(valFold)), 2, pTgt);
    [~, maxIdx] = max(tgtXVal * wTgt, [], 2);
    yPred = labels(maxIdx);
    evalObj = evaluate(cell2mat(dataSplit.tgtY(valFold)), yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test with tgt test set
    tgtXTst = extSparseDim(cell2mat(dataSplit.tgtX(tstFold)), 2, pTgt);
    [~, maxIdx] = max(tgtXTst * wTgt, [], 2);
    yPred = labels(maxIdx);
    evalObj = evaluate(cell2mat(dataSplit.tgtY(tstFold)), yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgTransMacroF1(i) = evalObj.macroF1;
    avgTransMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%tgt%%%%%%%%%%%%%%%%
    microF1List = zeros(size(cList,2), 1);
    %% train on tgt train set
    for c = cList
        fprintf('tunning on C:%f...,', c);
        model = train(cell2mat(dataSplit.tgtY(trnFolds)), cell2mat(dataSplit.tgtX(trnFolds)), ['-s 4 -c ',num2str(c), ' -q']);
        yPred = predict(cell2mat(dataSplit.tgtY(valFold)), cell2mat(dataSplit.tgtX(valFold)), model, '-q');
        evalObj = evaluate(cell2mat(dataSplit.tgtY(valFold)), yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        microF1List(cList == c) = evalObj.microF1;
    end

    %% find the best hyperparameter
    [bestF1, bestIdx] = max(microF1List(:));
    fprintf('best micro F1 is %f, best C is %f\n', bestF1, cList(bestIdx));
    
    %% test on tgt tst set
    model = train(cell2mat(dataSplit.tgtY(trnFolds)), cell2mat(dataSplit.tgtX(trnFolds)), ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
    yPred = predict(cell2mat(dataSplit.tgtY(tstFold)), cell2mat(dataSplit.tgtX(tstFold)), model, '-q');
    evalObj = evaluate(cell2mat(dataSplit.tgtY(tstFold)), yPred);
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
    fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));
    
    fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



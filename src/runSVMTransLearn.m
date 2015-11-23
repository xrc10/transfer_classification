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

avgMacroF1 = zeros(repTime, 1);
avgMicroF1 = zeros(repTime, 1);
addpath('../../liblinear/');

for i = 1:repTime
    
    dataSplit = randBiSplit(['../data/', srcFolder], ['../data/', tgtFolder], classes);
    labels = unique(dataSplit.srcYTrn);
    
    %% train on src train set
    fprintf('Training with SVM...,');
    
    cList = 10.^(-3);
    
    microF1List = zeros(size(cList,2), 1);
    
    for c = cList
        fprintf('tunning on C:%f...,', c);
        mod = train(dataSplit.srcYTrn, dataSplit.srcXTrn, ['-s 4 -c ',num2str(c), ' -q']);
        yPred = predict(dataSplit.srcYVal, dataSplit.srcXVal, mod, '-q');
        evalObj = evaluate(dataSplit.srcYVal, yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        microF1List(cList == c) = evalObj.microF1;
    end
    
    %% find the best hyperparameter
    [bestF1, bestIdx] = max(microF1List(:));
    fprintf('best micro F1 is %f, best C is %f\n', bestF1, cList(bestIdx));
    
    %% train with best hyperparameter
    mod = train(dataSplit.srcYTrn, dataSplit.srcXTrn, ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
    wSrc = (mod.w)';
    wSrc = [wSrc; zeros(pSrc - size(wSrc,1), size(wSrc,2))];
    wTgt = simM * wSrc;
    dataSplit.tgtXTst = extSparseDim(dataSplit.tgtXTst, 2, pTgt);
    [~, maxIdx] = max(dataSplit.tgtXTst * wTgt, [], 2);
    yPred = labels(maxIdx);
    evalObj = evaluate(dataSplit.tgtYTst, yPred);
    fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgMacroF1(i) = evalObj.macroF1;
    avgMicroF1(i) = evalObj.microF1;
    
end

fprintf('-------Overall--------\n');
fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));

fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


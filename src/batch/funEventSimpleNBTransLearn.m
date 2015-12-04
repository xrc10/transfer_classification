function eval = funEventSimpleNBTransLearn(simM)

rng(315);

%% read data

tgtFolder = 'TDT5_Chinese_event_dict_wordcount';
srcFolder = 'TDT5_English_event_dict_wordcount';
% srcFolder = 'TDT5_Chinese_event_dict_wordcount';
% tgtFolder = 'TDT5_English_event_dict_wordcount';

pSrc = 17327;
pTgt = 30000;

kFold = 5;

%% load simM
% load(simMPath, 'simM');
%% normalize simM that each row sum up to 1
n =  sum( simM, 2 );
n( n == 0 ) = 1;
fprintf('normalizing similarity matrix...\n');
simM = bsxfun( @rdivide, simM, n );

% if ~exist(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder]), 'dir')
%     mkdir(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder]));
% end
% logFile = fopen(fullfile('log/', mfilename, [srcFolder, '_TO_', tgtFolder], 'log.txt'), 'w');

fprintf('loading data...\n');

addpath('../../tool');

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

dataSplit = kFoldEventBiSplit(['../../data/', srcFolder], ['../../data/', tgtFolder], classes, kFold);

for i = 1:kFold
    
    valFold = i;
    tstFold = mod(i, kFold)+1;
    trnFolds = setdiff((1:kFold)', [valFold; tstFold]);
    
    %%%%%%%%%%%%%%%src%%%%%%%%%%%%%%%%
    
    %% train on src train set
    fprintf('Training with Naive Bayes...,');
    model = NaiveBayes.fit(cell2mat(dataSplit.srcX(trnFolds)), cell2mat(dataSplit.srcY(trnFolds)), 'Distribution', 'mn');
    %     mod = fitNaiveBayes(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    
    %% test on src val set
    yPred = predict(model, cell2mat(dataSplit.srcX(valFold)));
    evalObj = evaluate(cell2mat(dataSplit.srcY(valFold)), yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test on src tst set
    yPred = predict(model, cell2mat(dataSplit.srcX(tstFold)));
    evalObj = evaluate(cell2mat(dataSplit.srcY(tstFold)), yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgSrcMacroF1(i) = evalObj.macroF1;
    avgSrcMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%trans%%%%%%%%%%%%%%%%
    
    %% transfer the learned model
    labels = unique(cell2mat(dataSplit.srcY));
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
    tgtXVal = extSparseDim(cell2mat(dataSplit.tgtX(valFold)), 2, pTgt);
    yPred = myNBPredict(modTgt, tgtXVal);
    evalObj = evaluate(cell2mat(dataSplit.tgtY(valFold)), yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test with tgt test set
    tgtXTst = extSparseDim(cell2mat(dataSplit.tgtX(tstFold)), 2, pTgt);
    yPred = myNBPredict(modTgt, tgtXTst);
    evalObj = evaluate(cell2mat(dataSplit.tgtY(tstFold)), yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    avgTransMacroF1(i) = evalObj.macroF1;
    avgTransMicroF1(i) = evalObj.microF1;
    
    %%%%%%%%%%%%%%%tgt%%%%%%%%%%%%%%%%
    
    %% train on tgt train set
    fprintf('Training with Naive Bayes...,');
    model = NaiveBayes.fit(cell2mat(dataSplit.tgtX(trnFolds)), cell2mat(dataSplit.tgtY(trnFolds)), 'Distribution', 'mn');
    %     mod = fitNaiveBayes(dataSplit.srcXTrn, dataSplit.srcYTrn, 'Distribution', 'mn');
    
    %% test on tgt val set
    yPred = predict(model, cell2mat(dataSplit.tgtX(valFold)));
    evalObj = evaluate(cell2mat(dataSplit.tgtY(valFold)), yPred);
    fprintf('Validation: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    %% test on tgt tst set
    yPred = predict(model, cell2mat(dataSplit.tgtX(tstFold)));
    evalObj = evaluate(cell2mat(dataSplit.tgtY(tstFold)), yPred);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
    
    avgTgtMacroF1(i) = evalObj.macroF1;
    avgTgtMicroF1(i) = evalObj.microF1;
end

eval.avgSrcMacroF1 = mean(avgSrcMacroF1);
eval.avgTgtMacroF1 = mean(avgTgtMacroF1);
eval.avgTransMacroF1 = mean(avgTransMacroF1);
eval.avgSrcMicroF1 = mean(avgSrcMicroF1);
eval.avgTgtMicroF1 = mean(avgTgtMicroF1);
eval.avgTransMicroF1 = mean(avgTransMicroF1);


% aggMacroF1 = [avgSrcMacroF1, avgTgtMacroF1, avgTransMacroF1];
% aggMicroF1 = [avgSrcMicroF1, avgTgtMicroF1, avgTransMicroF1];
% tag = {'src', 'tgt', 'trans'};

% for i = 1:3
%     
%     avgMacroF1 = aggMacroF1(:,i);
%     avgMicroF1 = aggMicroF1(:,i);
%     
%     fprintf(['-------', tag{i}, ' Overall--------\n']);
%     fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
%     fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));
%     
%     fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
%     fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));
% 
% end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clc; clear all; close all;

%% read data

f1 = 'TDT5_Chinese_event_dict_wordcount';
f2 = 'TDT5_English_event_dict_wordcount';
f3 = 'TDT5_Chinese_event_wordcount';
f4 = 'TDT5_English_event_wordcount';
folder_names = {f1, f2, f3, f4};

repTime = 100;

for f = 1:4
    folder_name = folder_names{f};
    
    if ~exist(fullfile('log/', mfilename, folder_name), 'dir')
        mkdir(fullfile('log/', mfilename, folder_name));
    end
    logFile = fopen(fullfile('log/', mfilename, folder_name, 'log.txt'), 'w');
    
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
        
        dataSplit = randSplit(['../data/', folder_name], classes);
        
        cList = 10.^(-3:3);
        addpath('../../liblinear/');
        microF1List = zeros(size(cList,2), 1);
        for c = cList
            fprintf('tunning on C:%f...,', c);
            mod = train(dataSplit.yTrn, dataSplit.XTrn, ['-s 4 -c ',num2str(c), ' -q']);
            yPred = predict(dataSplit.yVal, dataSplit.XVal, mod, '-q');
            evalObj = evaluate(dataSplit.yVal, yPred);
            fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
            microF1List(cList == c) = evalObj.microF1;
        end
        
        %% find the best hyperparameter
        [bestF1, bestIdx] = max(microF1List(:));
        fprintf('best C is %f\n', cList(bestIdx));
        
        %% train with best hyperparameter
        mod = train(dataSplit.yTrn, dataSplit.XTrn, ['-s 4 -c ',num2str(cList(bestIdx)), ' -q']);
        yPred = predict(dataSplit.yTst, dataSplit.XTst, mod, '-q');
        evalObj = evaluate(dataSplit.yTst, yPred);
        fprintf('macro F1 is %f, micro F1 is %f\n', evalObj.macroF1, evalObj.microF1);
        avgMacroF1(i) = evalObj.macroF1;
        avgMicroF1(i) = evalObj.microF1;
    end
    
    fprintf('-------Overall--------\n');
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    fprintf('Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));
    
    fprintf(logFile, 'Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    fprintf(logFile, 'Test: CI of macro F1 is %f, CI of micro F1 is %f\n', 2*sqrt(var(avgMacroF1)/repTime), 2*sqrt(var(avgMicroF1)/repTime));
    
end


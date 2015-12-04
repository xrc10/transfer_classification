function batchRun(method, prob)

simSrc = 'top';
topLinkNum = 100;
simSrcFolder = ['../../data/linear_WE_transfer/', simSrc, 'SimMatCol', num2str(topLinkNum)];
dictSrc = 'dict';
dictSimSrcFolder = ['../../data/linear_WE_transfer/', dictSrc, 'SimMat'];

percts = dir(simSrcFolder);

% method = 'NB';
% method = 'SVM';
% methods = {'NB', 'SVM'};

% prob = 'Type';
% prob = 'Event';
% probs = {'Type', 'Event'};

if ~exist(fullfile('log', [simSrc, 'Col', num2str(topLinkNum), '_comp']), 'dir')
    mkdir(fullfile('log', [simSrc, 'Col', num2str(topLinkNum), '_comp']));
end

%% given method and problem
logFile = fopen(fullfile('log',[simSrc, 'Col', num2str(topLinkNum), '_comp'], [method, '_', prob, '.log']), 'w');

for i = 3:size(percts, 1)
    if percts(i).isdir
        %% given percts
        fprintf(logFile, '%s\t', percts(i).name);
        disp(['--', percts(i).name]);
        draws = dir(fullfile(simSrcFolder, percts(i).name));
        timeOfDraw = size(draws,1) - 2;
        avgSrcMacroF1 = zeros(timeOfDraw, 1);
        avgSrcMicroF1 = zeros(timeOfDraw, 1);
        avgTgtMacroF1 = zeros(timeOfDraw, 1);
        avgTgtMicroF1 = zeros(timeOfDraw, 1);
        avgTransMacroF1 = zeros(timeOfDraw, 1);
        avgTransMicroF1 = zeros(timeOfDraw, 1);
        for j = 3:size(draws,1)
            %% load simM from simSrc
            matFileName = fullfile(simSrcFolder, percts(i).name, draws(j).name);
            load(matFileName, 'simM');
            tmpSimM = simM;
            %% load groundtruth dict simM
            matFileName = fullfile(dictSimSrcFolder, percts(i).name, draws(j).name);
            load(matFileName, 'simM');
            %% merge two simM together
            tmpSimM(:,sum(simM,1)>0) = 0;
            simM = simM + tmpSimM;
            
            if strcmp(prob, 'Type') && strcmp(method, 'NB')
                eval = funTypeSimpleNBTransLearn(simM);
            elseif strcmp(prob, 'Type') && strcmp(method, 'SVM')
                eval = funTypeSimpleSVMTransLearn(simM);
            elseif strcmp(prob, 'Event') && strcmp(method, 'NB')
                eval = funEventSimpleNBTransLearn(simM);
            elseif strcmp(prob, 'Event') && strcmp(method, 'SVM')
                eval = funEventSimpleSVMTransLearn(simM);
            end
            avgSrcMacroF1(j-2) = eval.avgSrcMacroF1;
            avgSrcMicroF1(j-2) = eval.avgSrcMicroF1;
            avgTgtMacroF1(j-2) = eval.avgTgtMacroF1;
            avgTgtMicroF1(j-2) = eval.avgTgtMicroF1;
            avgTransMacroF1(j-2) = eval.avgTransMacroF1;
            avgTransMicroF1(j-2) = eval.avgTransMicroF1;
        end
        %% write to log
        aggMacroF1 = [avgSrcMacroF1, avgTgtMacroF1, avgTransMacroF1];
        aggMicroF1 = [avgSrcMicroF1, avgTgtMicroF1, avgTransMicroF1];
        tag = {'src', 'tgt', 'trans'};
        
        for j = 1:3
            
            avgMacroF1 = aggMacroF1(:,j);
            avgMicroF1 = aggMicroF1(:,j);
            
            fprintf(['-------', tag{j}, ' Overall--------\n']);
            fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
            
            fprintf(logFile, '%f\t%f\t', mean(avgMacroF1), mean(avgMicroF1));
        end
        fprintf(logFile, '\n');
    end
    
    
end




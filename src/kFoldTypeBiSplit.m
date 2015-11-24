function [ dataSplit ] = kFoldTypeBiSplit( srcInputDir, tgtInputDir, classNames, kFold )
%KFOLDTYPEBISPLIT Summary of this function goes here
%   Detailed explanation goes here

addpath('../tool');

XAll = cell(kFold, 2);
YAll = cell(kFold, 2);
for j = 1:2
    for i = 1:kFold
        XAll{i,j} = [];
        YAll{i,j} = [];
    end
end

inputDirs = {srcInputDir, tgtInputDir};

for j = 1:2
    inputDir = inputDirs{j};
    for i = 1:size(classNames, 2)
        dirList = dir(fullfile(inputDir, classNames{i}));
        %% extract all svm file names
        svmFiles = extractSVMFiles(dirList);
        %% permutation
        svmFiles = svmFiles(randperm(size(svmFiles, 2)));
        %% read
        X = [];
        for k = 1:size(svmFiles,2)
            [~, XTmp] = libsvmread(fullfile(inputDir, classNames{i}, svmFiles{k}));
            X = sparseVStack(X, XTmp);
        end
        y = i*ones(size(X,1), 1);
        %% permutation again
        randIdx = randperm(size(X, 1));
        X = X(randIdx,:);
        %% split
        foldSize = floor(size(X,1)/kFold);
        for k = 1:kFold
            XAll{k,j} = sparseVStack(XAll{k,j}, X((k-1)*foldSize + 1:k*foldSize,:));
            YAll{k,j} = [YAll{k,j}; y((k-1)*foldSize + 1:k*foldSize)];
        end
        for k = foldSize*kFold+1:size(X,1)
            XAll{mod(k, kFold)+1,j} = sparseVStack(XAll{mod(k, kFold)+1,j}, X(k,:));
            YAll{mod(k, kFold)+1,j} = [YAll{mod(k, kFold)+1,j}; y(k)];
        end
        
    end
end

dataSplit.srcX = XAll(:,1);
dataSplit.tgtX = XAll(:,2);
dataSplit.srcY = YAll(:,1);
dataSplit.tgtY = YAll(:,2);


end

function X = sparseVStack(X1, X2)
if size(X1, 2) == size(X2, 2)
    X = [X1; X2];
elseif size(X1, 2) > size(X2, 2)
    X = [X1; extSparseDim(X2, 2, size(X1,2))];
else
    X = [extSparseDim(X1 ,2, size(X2,2)); X2];
end
end

function svmFiles = extractSVMFiles(dirList)
svmFiles = {};
for j = 1:size(dirList, 1)
    if strendswith(dirList(j).name, '.svm')
        svmFiles = [svmFiles, dirList(j).name];
    end
end
end

function b = strendswith(s, pat)
%STRENDSWITH Determines whether a string ends with a specified pattern
%
%   b = strstartswith(s, pat);
%       returns whether the string s ends with a sub-string pat.
%

%   History
%   -------
%       - Created by Dahua Lin, on Oct 9, 2008
%

%% main

sl = length(s);
pl = length(pat);

b = (sl >= pl && strcmp(s(sl-pl+1:sl), pat)) || isempty(pat);

end


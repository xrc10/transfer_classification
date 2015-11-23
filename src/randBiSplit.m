function [ dataSplit ] = randBiSplit( srcInputDir, tgtInputDir, classNames )
%RANDSPLIT Summary of this function goes here
%   randomly split dataset without letting same event appears in both
%   training and testing in both src and tgt split
addpath('../tool');

srcYTrn = [];
srcYVal = [];
srcYTst = [];
srcXTrn = [];
srcXVal = [];
srcXTst = [];

tgtYTrn = [];
tgtYVal = [];
tgtYTst = [];
tgtXTrn = [];
tgtXVal = [];
tgtXTst = [];

for i = 1:size(classNames, 2)
    srcDirList = dir(fullfile(srcInputDir, classNames{i}));
    tgtDirList = dir(fullfile(tgtInputDir, classNames{i}));
    %% extract all svm file names
    srcSvmFiles = extractSVMFiles(srcDirList);
    tgtSvmFiles = extractSVMFiles(tgtDirList);
    
    %% random extract train, validation and test split
    %% make sure tgt val and tst not appear in src train
    randTgtSvmFiles = tgtSvmFiles(randperm(size(tgtSvmFiles, 2)));
    srcTrnsSvmPool = setdiff(srcSvmFiles, randTgtSvmFiles(end-3:end));
    srcRemainSvmPool = setdiff(srcSvmFiles, srcTrnsSvmPool);
    randSrcSvmFiles = [srcTrnsSvmPool(randperm(size(srcTrnsSvmPool, 2))),...
        srcRemainSvmPool(randperm(size(srcRemainSvmPool, 2)))];
    %% find out intersection
    insctSvmFiles = intersect(randSrcSvmFiles(1:end-4), randTgtSvmFiles(end-3:end));
    if size(insctSvmFiles, 2) > 0
        disp('Wrong Split!!!');
    end
    
    [XTrn, XVal, XTst, yTrn, yVal, yTst] = readSplitData(srcInputDir, classNames{i}, i, randSrcSvmFiles);
    srcYTrn = [srcYTrn; yTrn];
    srcYVal = [srcYVal; yVal];
    srcYTst = [srcYTst; yTst];
    srcXTrn = sparseVStack(srcXTrn, XTrn);
    srcXVal = sparseVStack(srcXVal, XVal);
    srcXTst = sparseVStack(srcXTst, XTst);
    [XTrn, XVal, XTst, yTrn, yVal, yTst] = readSplitData(tgtInputDir, classNames{i}, i, randTgtSvmFiles);
    tgtYTrn = [tgtYTrn; yTrn];
    tgtYVal = [tgtYVal; yVal];
    tgtYTst = [tgtYTst; yTst];
    tgtXTrn = sparseVStack(tgtXTrn, XTrn);
    tgtXVal = sparseVStack(tgtXVal, XVal);
    tgtXTst = sparseVStack(tgtXTst, XTst);
end

pSrc = max([size(srcXTrn,2), size(srcXVal,2), size(srcXTst,2)]);
pTgt = max([size(tgtXTrn,2), size(tgtXVal,2), size(tgtXTst,2)]);

srcXTrn = extSparseDim(srcXTrn, 2, pSrc);
srcXVal = extSparseDim(srcXVal, 2, pSrc);
srcXTst = extSparseDim(srcXTst, 2, pSrc);
tgtXTrn = extSparseDim(tgtXTrn, 2, pTgt);
tgtXVal = extSparseDim(tgtXVal, 2, pTgt);
tgtXTst = extSparseDim(tgtXTst, 2, pTgt);

dataSplit.srcXTrn = srcXTrn;
dataSplit.srcXVal = srcXVal;
dataSplit.srcXTst = srcXTst;
dataSplit.srcYTrn = srcYTrn;
dataSplit.srcYVal = srcYVal;
dataSplit.srcYTst = srcYTst;

dataSplit.tgtXTrn = tgtXTrn;
dataSplit.tgtXVal = tgtXVal;
dataSplit.tgtXTst = tgtXTst;
dataSplit.tgtYTrn = tgtYTrn;
dataSplit.tgtYVal = tgtYVal;
dataSplit.tgtYTst = tgtYTst;

end

function [XTrn, XVal, XTst, yTrn, yVal, yTst] = readSplitData(inputDir, className, classIdx, randSvmFiles)
yTrn = [];
yVal = [];
yTst = [];
XTrn = [];
XVal = [];
XTst = [];
%% trn
for j = 1:size(randSvmFiles,2)-4
    [~, X] = libsvmread(fullfile(inputDir, className, randSvmFiles{j}));
    y = classIdx * ones(size(X,1), 1);
    XTrn = sparseVStack(XTrn, X);
    yTrn = [yTrn; y];
end
%% tst
for j = size(randSvmFiles,2)-3:size(randSvmFiles,2)-2
    [~, X] = libsvmread(fullfile(inputDir, className, randSvmFiles{j}));
    y = classIdx * ones(size(X,1), 1);
    XTst = sparseVStack(XTst, X);
    yTst = [yTst; y];
end
%% val
for j = size(randSvmFiles,2)-1:size(randSvmFiles,2)
    [~, X] = libsvmread(fullfile(inputDir, className, randSvmFiles{j}));
    y = classIdx * ones(size(X,1), 1);
    XVal = sparseVStack(XVal, X);
    yVal = [yVal; y];
end
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
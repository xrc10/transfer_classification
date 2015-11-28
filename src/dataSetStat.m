function dataSetStat

addpath('../tool');

tgtInputDir = '../data/TDT5_Chinese_event_dict_wordcount';
srcInputDir = '../data/TDT5_English_event_dict_wordcount';

classNames = {'Accidents',...
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

for i = 1:size(classNames, 2)
    srcDirList = dir(fullfile(srcInputDir, classNames{i}));
    tgtDirList = dir(fullfile(tgtInputDir, classNames{i}));
    %% extract all svm file names
    srcSvmFiles = extractSVMFiles(srcDirList);
    tgtSvmFiles = extractSVMFiles(tgtDirList);
    overlapSvmFiles = intersect(srcSvmFiles, tgtSvmFiles);
    fprintf('%s\t%d\t%d\t%d\n', classNames{i}, size(srcSvmFiles, 2), size(tgtSvmFiles, 2), size(overlapSvmFiles,2));
    for j = 1:size(srcSvmFiles, 2)
        fprintf('%d ', countDoc(fullfile(srcInputDir, classNames{i}, srcSvmFiles{j})));
    end
    fprintf('\t');
    for j = 1:size(tgtSvmFiles, 2)
        fprintf('%d ', countDoc(fullfile(tgtInputDir, classNames{i}, tgtSvmFiles{j})));
    end
    fprintf('\t');
    for j = 1:size(overlapSvmFiles, 2)
        fprintf('%d ', countDoc(fullfile(srcInputDir, classNames{i}, overlapSvmFiles{j})));
    end
    fprintf('\t');
    for j = 1:size(overlapSvmFiles, 2)
        fprintf('%d ', countDoc(fullfile(tgtInputDir, classNames{i}, overlapSvmFiles{j})));
    end
    fprintf('\n');
end

end

function c = countDoc(p)
[~, X] = libsvmread(p);
c = size(X,1);
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
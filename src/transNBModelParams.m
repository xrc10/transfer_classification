function [ paramsTgt ] = transNBModelParams( paramsSrc, simM )
%TRANSMODELPARAMS Summary of this function goes here
% paramsSrc: class_num * src_word_dim matrix
% paramsTgt: class_num * tgt_word_dim matrix
% simM: src_word_dim * tgt_word_dim matrix

%% normalize simM that each row sum up to 1
% n =  sum( simM, 2 );
% n( n == 0 ) = 1;
% fprintf('normalizing similarity matrix...\n');
% simM = bsxfun( @rdivide, simM, n );
%% simple weighted summation
fprintf('transfering NB model...\n');
paramsTgt = paramsSrc * simM;

%% one way to remove zeros in paramsTgt
% paramsTgt = paramsTgt + 1/size(paramsTgt, 2);

%% another way to remove zeros in paramsTgt


%% normalize paramsTgt
n =  sum( paramsTgt, 2 );
n( n == 0 ) = 1;
fprintf('normalizing parameter matrix...\n');
paramsTgt = bsxfun( @rdivide, paramsTgt, n );

end


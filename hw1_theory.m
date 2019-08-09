%% CS 383
% Alan Davis
% Assignment 1 - Part 1

%% Clear All
clear, clc, close all

%% Defining Classes
% Class 1 and 2 are the datasets used to find the information gains in the
% code.
class1 = [-2 1; -5 -4; -3 1; 0 3; -8 11];
class2 = [-2 5; 1 0; 5 -1; -1 -3; 6 1];

%% Feature Creation & Unique Feature Values
% f1 and f2 are the two features which were made by stacking their
% respective columns of numbers from the original classes
f1 = [class1(:,1); class2(:,1)];
f2 = [class1(:,2); class2(:,2)];

% featlen is just the total length of each feature set
featlen = size(f1,1);

% vals1 and vals2 hold the unique vaules of feature sets 1 and 2 (f1 and
% f2)
vals1 = unique(f1);
vals2 = unique(f2);

%% Probability Matrices
% probf1 and probf2 count how many times a unique number shows up in there
% respective columns of f1 and f2. It simply does this by calling the
% function prob (see below for equations).
probf1 = prob(class1, class2, 1, vals1);
probf2 = prob(class1, class2, 2, vals2);

%% Finding Entropy
%
% $$Entropy=E=\sum frac1*(frac2+frac3)$$

% entropy1 and entropy2 represent the entropys of class1 and class2. It
% calls the entr function (see below for equations).
entropy1 = entr(probf1, featlen);
entropy2 = entr(probf2, featlen);

%% Information Gain 
%
% $$Information Gain = IG = 1-E =1-\sum frac1*(frac2+frac3)$$

% ig1 and ig2 are the respective information gains for class1 and class2
% respectively.
ig1 = 1 - entropy1;
ig2 = 1 - entropy2;

%% Functions
%%% Entropy Function
% The following function calculates entropy for any given feature set as
% long as a 2 column matrix is provided with the positive values on the 
% left side and the negative values on the right, and the total number of 
% values in the feature set is given. The function will then begin to pick
% apart the given data, and calculate the final entropy for the matrix
% provided.
%
% $$Entropy=E=\sum frac1*(frac2+frac3)$$
%
% $$frac_1=\frac{v_1+v_2}{len}$$
%
% $$frac_2=\frac{-v_1}{v_1+v_2}*log_2(\frac{v_1}{v_1+v_2})$$
%
% $$frac_3=\frac{-v_2}{v_1+v_2}*log_2(\frac{v_2}{v_1+v_2})$$
function entropy = entr(mat, len)
    % Create an empty matrix to hold all of the summations
    ansmat = [];
    % For every row inside of the feature probability matrix pull the first
    % columns value, and the second setting it to v1 and v2 respectively.
    for o = 1:size(mat,1)
        v1 = mat(o,1); v2 = mat(o,2);
        % If v1 or v2 is "0" make them both zero to not return "NaN" values
        if v1 == 0 || v2 == 0
           v1 = 0;
           v2 = 0;
        else
            v1 = mat(o,1); v2 = mat(o,2);
        end
        
        % Definining fraction 1 as stated above, and quickly defining the
        % denomenators for frac2 and frac3 to a variable (denom).
        frac1 = (v1+v2)/len;
        denom = v1+v2;
        
        % If v1 or v2 hold the value "0" then zero out the fraction so
        % their fraction so that log2 does not return "NaN"
        if v1 == 0
            frac2 = 0;
        else
            frac2 = (-v1/denom)*log2(v1/denom);
        end
        if v2 == 0
            frac3 = 0;
        else
            frac3 = (-v2/denom)*log2(v2/denom);
        end
        % Grab the answer and append it to the ansmat matrix.
        ans = frac1*(frac2+frac3);
        ansmat(end+1,:) = ans;
    end
    entropy = sum(ansmat);
end

%%% Probability Matrix Funtion
% The following function quickly checks any features matrix with the unique
% value of that matrix. If the a unique value is in the column it is
% checking, it will add 1 to the count of that number. To do this you need
% both classes you are pulling feature values from, the column number that
% the features are in, and an array of all of the unique values.
function probmat = prob(class1, class2, colnum, vals)
    % Creates two empty columns to hold the counts for each value per
    % column.
    newcol1 = [];
    newcol2 = [];
    % Iterate only the amount of times that the class has columns.
    for i=1:size(class1,2)
        % if i equals 1 check the numbers in the first column for unique
        % number matches. Otherwise, do the same for the second column.
        if i == 1
            col = class1(:,colnum);
        else
            col = class2(:,colnum);
        end
        % For every value of the unique numbers of the feature check to see
        % if the column contains that number. If it has the number add 1 to
        % the count.
        for j=1:size(vals,1)
           if i == 1
            newcol1(end+1,:) = nnz(col == vals(j,:));
           elseif i > 1
              newcol2(end+1,:) = nnz(col == vals(j,:));
           end
       end
    end
    % Append the two columns together so that it is a two column matrix.
    probmat = [newcol1 newcol2];
end
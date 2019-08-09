%% CS 383
% Alan Davis
% Assignment 1 - Part 2 & 3

%% Clear All
clc
clear, close all

%% Creating matrix
% Create empty matrix to hold all of the images in rows

finmat = [];

% Allows for a "for loop" to iterate through the files in the yalefaces
% directory.

files = dir('yalefaces/*');

% This for loop iterates through the yalefaces directory, pulls the most
% current image and reads it into matlab as a 1x1600 array, and reshapes
% that array to be a 40x40 matrix. That 40x40 matrix is then appended as a
% column onto the finmat matrix.

for z = 1:length(files)
    if contains(files(z).name, 'subject')
        baseFileName = files(z).name;
        fullFileName = fullfile('yalefaces', files(z).name);
        imdata = imread(fullFileName);
        subsampled = imresize(imdata,[40,40]);
        finmat(end+1,:) = subsampled(:);
    end
end

%% Standardizing data
% Simply standardizes the data in preparation of performing PCA on the
% data.

m = mean(finmat);
s = std(finmat);

finmat = finmat - repmat(m,size(finmat,1),1);
finmat = finmat ./ repmat(s,size(finmat,1),1);

%% 2D using PCA
% Built-in function that calculates covariance.

C = cov(finmat);

% Built-in function that calculates both eigenvectors and eigenvalues from 
% the covariance matrix C, and places them into matrices "V" and "D" 
% respectively.

[V, D] = eig(C);

% Duplicates the eigenvalues for later use

D2 = D;

% Takes the largest eigenvalues and sorts them in one row from largest to
% smallest.

maxvals = fliplr(max(D));

% Creates an empty array to hold "k" amount of the biggest eigenvalues.

arrTop = [];

% sumBottom is equal to the sum of all the eigenvalues in D

sumBottom = sum(sum(abs(D)));

% For every single column in D take the first row, column "y" value from
% maxvals, and store the value in "val". Append that value to matrix
% "arrTop". Sum all of the values in arrTop and store the number in
% "sumTop". If "sumTop/sumBottom" is greater than or equal to 0.95 set k
% equal to the current iteration number ("y").

for y = 1:size(D,2)
    val = maxvals(1,y);
    arrTop(:, end+1) = val;
    sumTop = sum(arrTop);
    if sumTop/sumBottom >= 0.95
        k = y;
        break
    end
end

% Make an empty matrix to hold the data needed to make an eigenface.

Zmat = [];

% Take the first two rows from finmat and multiply them by the two 
% eigenvectors that have the largest eigenvalues correlating to them. (Zero
% out each column that you find the max to be after each iteration).

for z = 1:2
   [~, col] = find(D == max(max(D)));
   Z = finmat*V(:,col);
   Zmat(:,end+1) = Z;
   D(:,col) = 0;
end

%% Plot
% Plot Zmat columns 1 and 2 to get a PCA plot of the data.

figure;
title('2D PCA Projection of Data');
axes('LineWidth',0.6,...
    'FontName','Helvetica',...
    'FontSize',8);
line(Zmat(:,1),Zmat(:,2),...
    'LineStyle','None',...
    'Marker','o');
grid on

%% Original Image
% The first row of finmat holds 'subject02.centerlight'
% Reshape the image to be 40x40 and display the image to scale and
% grayscaled.

originalImage = finmat(1,:);
smallerImage = reshape(originalImage,[40,40]);

figure; 
set(gcf,'colormap',gray); 
image(smallerImage, 'CDataMapping', 'scaled'); 
title('Original Image');

%% Eigenface
% Grab the max value from the duplicate of the eigenvalues matrix "D" 
% column and take the respective columns data. Resize it to a 40x40 image.

[~, eigcol] = max(diag(D2));
eigarr = V(:,eigcol);
resized = -reshape(eigarr,40,40);

figure;
set(gcf,'colormap',gray);
image(resized, 'CDataMapping', 'scaled');
title('Primary Principle Component');

%% Reconstruction
% wvals is "k" eigenvectors that correlate to the "k" largest eigenvalues.

wvals = V(:,(1600-k):1600);

% z is wvals multiplied by the first image 'subject02.centerlight'.

z = finmat(1,:)*wvals;

% finimage is the 40x40 version of z multiplied by the transposed version
% of matrix "wvals"

finimage = reshape(z*transpose(wvals),[40, 40]);

% Displays the image to scale and grayscales it.

figure;
set(gcf,'colormap',gray);
image(finimage, 'CDataMapping', 'scaled');
title('k PC Reconstruction');

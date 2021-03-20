clear; close all; clc

%% Read in data
[trnImages, trnLabels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[testImages, testLabels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

%% Create data matrices and compute SVD of training data
% Convert from uint8 to double
trnImages = im2double(trnImages);
testImages = im2double(testImages);

% Create training data matrix
dim = size(trnImages,1)*size(trnImages,2);
numImages = size(trnImages,3);
dataMat = zeros(dim,numImages);
for j = 1:numImages
    img = trnImages(:,:,j);
    imgCol = reshape(img,[dim,1]);
    dataMat(:,j) = imgCol;
end

% Create test data matrix
numTestImages = size(testImages,3);
testDataMat = zeros(dim,numTestImages);
for j = 1:numTestImages
    img = testImages(:,:,j);
    imgCol = reshape(img,[dim,1]);
    testDataMat(:,j) = imgCol;
end

% Demean data
for k = 1:dim
   feature = dataMat(k,:);
   dataMat(k,:) = feature - mean(feature);
   testDataMat(k,:) = testDataMat(k,:) - mean(feature);
end

% Compute SVD
[U,S,V] = svd(dataMat, 'econ');

%% Calculate and plot sigma energies
% Calculate sigma energies
sig = diag(S);
energies = sig.^2/sum(sig.^2);

% Create plot
plot(energies,'ko','Linewidth',2)
title('Sigma Energies')
xlabel('Singular Values')
ylabel('Fraction of Total Energy')

%% Plot first six principal components
figure
for k = 1:6
   subplot(1,6,k)
   ut1 = reshape(U(:,k),28,28);
   ut2 = rescale(ut1);
   imshow(ut2)
   title(['Eigendigit ',num2str(k)])
end
sgtitle('Principal Components of Training Data')

%% Create color-coded projection plot of data
% Calculate data projections
UT = U';
proj = UT(1:3,:)*dataMat;

% Find which data points correspond to which digits
zerosIdx = find(trnLabels == 0);
onesIdx = find(trnLabels == 1);
twosIdx = find(trnLabels == 2);
threesIdx = find(trnLabels == 3);
foursIdx = find(trnLabels == 4);
fivesIdx = find(trnLabels == 5);
sixesIdx = find(trnLabels == 6);
sevensIdx = find(trnLabels == 7);
eightsIdx = find(trnLabels == 8);
ninesIdx = find(trnLabels == 9);

% Plot color-coded projection, rainbow order
figure
plot3(proj(1,zerosIdx),proj(2,zerosIdx),proj(3,zerosIdx),'Marker','o','LineStyle','none','Color','#A2142F');
hold on
plot3(proj(1,onesIdx),proj(2,onesIdx),proj(3,onesIdx),'ro');
plot3(proj(1,twosIdx),proj(2,twosIdx),proj(3,twosIdx),'Marker','o','LineStyle','none','Color','#D95319');
plot3(proj(1,threesIdx),proj(2,threesIdx),proj(3,threesIdx),'Marker','o','LineStyle','none','Color','#EDB120');
plot3(proj(1,foursIdx),proj(2,foursIdx),proj(3,foursIdx),'go');
plot3(proj(1,fivesIdx),proj(2,fivesIdx),proj(3,fivesIdx),'Marker','o','LineStyle','none','Color','#77AC30');
plot3(proj(1,sixesIdx),proj(2,sixesIdx),proj(3,sixesIdx),'co');
plot3(proj(1,sevensIdx),proj(2,sevensIdx),proj(3,sevensIdx),'bo');
plot3(proj(1,eightsIdx),proj(2,eightsIdx),proj(3,eightsIdx),'Marker','o','LineStyle','none','Color','#7E2F8E');
plot3(proj(1,ninesIdx),proj(2,ninesIdx),proj(3,ninesIdx),'mo');
title('Data Projection')
xlabel('Mode 1')
ylabel('Mode 2')
zlabel('Mode 3')
legend('0','1','2','3','4','5','6','7','8','9')

%% Calculate variables to be used later
projFull = UT*dataMat; % Calculate full projection
rank = 50; % Use rank observed from sigma energies

%% Build LDA to classify 0s and 1s
numOnes = size(onesIdx,1); % Find number of ones
numZeros = size(zerosIdx,1); % Find number of sevens
ones = projFull(1:rank,onesIdx);
zeros = projFull(1:rank,zerosIdx);

% Calculate scatter matrices
meanOnes = mean(ones,2);
meanZeros = mean(zeros,2);

wnClassVar = 0; % within class variances
for k = 1:numOnes
    wnClassVar = wnClassVar + (ones(:,k) - meanOnes)*(ones(:,k) - meanOnes)';
end

for k = 1:numZeros
   wnClassVar =  wnClassVar + (zeros(:,k) - meanZeros)*(zeros(:,k) - meanZeros)';
end

bwClassVar = (meanOnes-meanZeros)*(meanOnes-meanZeros)'; % between class variance

% Find best projection line
[V2, D] = eig(bwClassVar,wnClassVar); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w1 = V2(:,ind);
w1 = w1/norm(w1,2);

% Project onto w
vOnes = w1'*ones;
vZeros = w1'*zeros;

% Make ones below the threshold
if mean(vOnes) > mean(vZeros)
    w1 = -w1;
    vOnes = -vOnes;
    vZeros = -vZeros;
end

% Find the threshold value
sortOnes = sort(vOnes);
sortZeros = sort(vZeros);

t1 = length(sortOnes);
t2 = 1;
while sortOnes(t1) > sortZeros(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
thresholdOnesZeros = (sortOnes(t1) + sortZeros(t2))/2;

% Plot histogram of results
figure
subplot(1,2,1)
histogram(sortOnes,30); hold on, plot([thresholdOnesZeros thresholdOnesZeros],[0 1200],'r')
set(gca,'Xlim',[-5 2],'Fontsize',14)
title('Ones')
xlabel('Ones Data Projection onto w')
ylabel('Number of Data Points in Bin')
subplot(1,2,2)
histogram(sortZeros,30); hold on, plot([thresholdOnesZeros thresholdOnesZeros],[0 1000],'r')
set(gca,'Xlim',[-2 5],'Fontsize',14)
title('Zeros')
xlabel('Zeros Data Projection onto w')
ylabel('Number of Data Points in Bin')

%% Build LDA to classify (0s and 1s) and 3s
onesZerosIdx = find(trnLabels <= 1);
numOnesZeros = size(onesZerosIdx,1); % Find number of (ones and zeros)
numThrees = size(threesIdx,1); % Find number of threes
onesZeros = projFull(1:rank,onesZerosIdx);
threes = projFull(1:rank,threesIdx);

% Calculate scatter matrices
meanOnesZeros = mean(onesZeros,2);
meanThrees = mean(threes,2);

wnClassVar = 0; % within class variances
for k = 1:numOnesZeros
    wnClassVar = wnClassVar + (onesZeros(:,k) - meanOnesZeros)*(onesZeros(:,k) - meanOnesZeros)';
end

for k = 1:numThrees
   wnClassVar =  wnClassVar + (threes(:,k) - meanThrees)*(threes(:,k) - meanThrees)';
end

bwClassVar = (meanOnesZeros-meanThrees)*(meanOnesZeros-meanThrees)'; % between class variance

% Find best projection line
[V2, D] = eig(bwClassVar,wnClassVar); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w2 = V2(:,ind);
w2 = w2/norm(w2,2);

% Project onto w
vOnesZeros = w2'*onesZeros;
vThrees = w2'*threes;

% Make threes above the threshold
if mean(vOnesZeros) > mean(vThrees)
    w2 = -w2;
    vOnesZeros = -vOnesZeros;
    vThrees = -vThrees;
end

% Find the threshold value
sortOnesZeros = sort(vOnesZeros);
sortThrees = sort(vThrees);

t1 = length(sortOnesZeros);
t2 = 1;
while sortOnesZeros(t1) > sortThrees(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
thresholdThrees = (sortOnesZeros(t1) + sortThrees(t2))/2;

% Plot histogram of results
figure
subplot(2,2,1)
histogram(sortOnesZeros,30); hold on, plot([thresholdThrees thresholdThrees],[0 3300],'r')
set(gca,'Xlim',[-4 4],'Fontsize',14)
title('Ones and Zeros')
xlabel('Ones and Zeros Data Projection onto w')
ylabel('Number of Data Points in Bin')
ylim([0 3300])
subplot(2,2,2)
histogram(sortThrees,30); hold on, plot([thresholdThrees thresholdThrees],[0 800],'r')
set(gca,'Xlim',[-3 5],'Fontsize',14)
title('Threes')
xlabel('Threes Data Projection onto w')
ylabel('Number of Data Points in Bin')

zerosNewIdx = [];
onesNewIdx = [];
for j = 1:numOnesZeros
    %if below threshold -> ones and zeros
    if vOnesZeros(j) < thresholdThrees
        % relate to index for ones/zeros to determine if one or zero
        foundInZerosIdx = find(zerosIdx==onesZerosIdx(j));
        % if empty, it's a one; otherwise, it's a zero
        % add index of zero or one to array storing indices for each
        if isempty(foundInZerosIdx) == 1
            foundInOnesIdx = find(onesIdx==onesZerosIdx(j));
            onesNewIdx(end+1) = onesIdx(foundInOnesIdx);
        else
            zerosNewIdx(end+1) = zerosIdx(foundInZerosIdx);
        end
    end
end
        
% find ones and zeros using indices
onesNew = projFull(1:rank,onesNewIdx);
zerosNew = projFull(1:rank,zerosNewIdx);

% project onto w from ones and zeros classifier
vOnesNew = w1'*onesNew;
vZerosNew = w1'*zerosNew;

% sort result
sortOnesNew = sort(vOnesNew);
sortZerosNew = sort(vZerosNew);

% Plot histogram of results
subplot(2,2,3)
histogram(sortOnesNew,30); hold on, plot([thresholdOnesZeros thresholdOnesZeros],[0 1200],'r')
set(gca,'Xlim',[-4 3],'Fontsize',14)
ylim([0 1200])
title('Ones')
xlabel('Ones Data Projection onto w')
ylabel('Number of Data Points in Bin')
subplot(2,2,4)
histogram(sortZerosNew,30); hold on, plot([thresholdOnesZeros thresholdOnesZeros],[0 1000],'r')
set(gca,'Xlim',[-2 5],'Fontsize',14)
title('Zeros')
xlabel('Zeros Data Projection onto w')
ylabel('Number of Data Points in Bin')

%% Build LDA to classify 4s and 9s, potentially most difficult to separate
numFours = size(foursIdx,1); % Find number of ones
numNines = size(ninesIdx,1); % Find number of sevens
fours = projFull(1:rank,foursIdx);
nines = projFull(1:rank,ninesIdx);

% Calculate scatter matrices
meanFours = mean(fours,2);
meanNines = mean(nines,2);

wnClassVar = 0; % within class variances
for k = 1:numFours
    wnClassVar = wnClassVar + (fours(:,k) - meanFours)*(fours(:,k) - meanFours)';
end

for k = 1:numNines
   wnClassVar =  wnClassVar + (nines(:,k) - meanNines)*(nines(:,k) - meanNines)';
end

bwClassVar = (meanFours-meanNines)*(meanFours-meanNines)'; % between class variance

% Find best projection line
[V2, D] = eig(bwClassVar,wnClassVar); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w3 = V2(:,ind);
w3 = w3/norm(w3,2);

% Project onto w
vFours = w3'*fours;
vNines = w3'*nines;

% Make fours below the threshold
if mean(vFours) > mean(vNines)
    w3 = -w3;
    vFours = -vFours;
    vNines = -vNines;
end

% Find the threshold value
sortFours = sort(vFours);
sortNines = sort(vNines);

t1 = length(sortFours);
t2 = 1;
while sortFours(t1) > sortNines(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
thresholdFoursNines = (sortFours(t1) + sortNines(t2))/2;

% Plot histogram of results
figure
subplot(1,2,1)
histogram(sortFours,30); hold on, plot([thresholdFoursNines thresholdFoursNines],[0 800],'r')
set(gca,'Xlim',[-4 3],'Fontsize',14)
title('Fours')
xlabel('Fours Data Projection onto w')
ylabel('Number of Data Points in Bin')
subplot(1,2,2)
histogram(sortNines,30); hold on, plot([thresholdFoursNines thresholdFoursNines],[0 1000],'r')
set(gca,'Xlim',[-3 4],'Fontsize',14)
title('Nines')
xlabel('Nines Data Projection onto w')
ylabel('Number of Data Points in Bin')

%% Quantify accuracy of LDA on test data for 4s and 9s
% Classify test data
testMat = UT*testDataMat; % PCA projection
foursTestIdx = find(testLabels == 4);
ninesTestIdx = find(testLabels == 9);
foursNinesTestIdx = cat(1,foursTestIdx,ninesTestIdx);
pval = w3'*testMat(1:rank,foursNinesTestIdx);

% Check pval against threshold
% nine = 1, four = 0
resVec = (pval > thresholdFoursNines);

% Checking performance
foursNinesLabels = 0*foursNinesTestIdx';
foursNinesLabels(length(foursTestIdx)+1:end) = 1;

% 0s are correct and 1s are incorrect
err = abs(resVec - foursNinesLabels);
errNum = sum(err);
foursNinesSucRate = 1 - errNum/numTestImages;

%% Quantify accuracy of LDA on test data for 0s and 1s
% Classify test data
zerosTestIdx = find(testLabels == 0);
onesTestIdx = find(testLabels == 1);
onesZerosTestIdx = cat(1,onesTestIdx,zerosTestIdx);
pval = w1'*testMat(1:rank,onesZerosTestIdx); % w1, testMat from before

% Check pval against threshold
% zero = 1, one = 0
resVec = (pval > thresholdOnesZeros);

% Checking performance
onesZerosLabels = 0*onesZerosTestIdx';
onesZerosLabels(length(onesTestIdx)+1:end) = 1;

% 0s are correct and 1s are incorrect
err = abs(resVec - onesZerosLabels);
errNum = sum(err);
onesZerosSucRate = 1 - errNum/numTestImages;

%% Create classification tree for all 10 digits
allTrainData = projFull(1:rank,:);
allTestData = testMat(1:rank,:);
tree = fitctree(allTrainData',trnLabels);
predTestLabels = predict(tree,allTestData');

% Calculate accuracy
err = abs(predTestLabels - testLabels);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
allTreeSucRate = 1 - errNum/numTestImages;

%% Create SVM classifier for all 10 digits
svm = fitcecoc(allTrainData',trnLabels);
predTestLabels = predict(svm,allTestData');

% Calculate accuracy
err = abs(predTestLabels - testLabels);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
allSVMSucRate = 1 - errNum/numTestImages;

%% Create classification tree for 4's and 9's
foursNinesIdx = cat(1,foursIdx,ninesIdx);
foursNinesTrain = projFull(1:rank,foursNinesIdx);
foursNinesTrainLabs = 0*foursNinesIdx';
foursNinesTrainLabs(1:length(foursIdx)) = 4;
foursNinesTrainLabs(length(foursIdx)+1:end) = 9;

foursNinesTest = testMat(1:rank,foursNinesTestIdx);
foursNinesTestLabs = 0*foursNinesTestIdx';
foursNinesTestLabs(1:length(foursTestIdx)) = 4;
foursNinesTestLabs(length(foursTestIdx)+1:end) = 9;

% Create tree
foursNinesTree = fitctree(foursNinesTrain',foursNinesTrainLabs);
predFoursNinesTestLabs = predict(tree,foursNinesTest');

% Calculate accuracy
err = abs(predFoursNinesTestLabs' - foursNinesTestLabs);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
foursNinesTreeSucRate = 1 - errNum/numTestImages;

%% Create SVM classifier for 4's and 9's
foursNinesSVM = fitcsvm(foursNinesTrain',foursNinesTrainLabs);
predFoursNinesTestLabs = predict(foursNinesSVM,foursNinesTest');

% Calculate accuracy
err = abs(predFoursNinesTestLabs' - foursNinesTestLabs);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
foursNinesSVMSucRate = 1 - errNum/numTestImages;

%% Create classification tree for 1's and 0's
onesZerosTrain = projFull(1:rank,onesZerosIdx);
onesZerosTrainLabs = 0*onesZerosIdx';
onesZerosTrainLabs(1:length(onesIdx)) = 1;
onesZerosTrainLabs(length(onesIdx)+1:end) = 0;

onesZerosTest = testMat(1:rank,onesZerosTestIdx);
onesZerosTestLabs = 0*onesZerosTestIdx';
onesZerosTestLabs(1:length(onesTestIdx)) = 1;
onesZerosTestLabs(length(onesTestIdx)+1:end) = 0;

% Create tree
onesZerosTree = fitctree(onesZerosTrain',onesZerosTrainLabs);
predOnesZerosTestLabs = predict(tree,onesZerosTest');

% Calculate accuracy
err = abs(predOnesZerosTestLabs' - onesZerosTestLabs);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
onesZerosTreeSucRate = 1 - errNum/numTestImages;

%% Create SVM classifier for 1's and 0's
onesZerosSVM = fitcsvm(onesZerosTrain',onesZerosTrainLabs);
predOnesZerosTestLabs = predict(onesZerosSVM,onesZerosTest');

% Calculate accuracy
err = abs(predOnesZerosTestLabs' - onesZerosTestLabs);
falseIdx = find(err ~= 0);
err(falseIdx) = 1;
errNum = sum(err);
onesZerosSVMSucRate = 1 - errNum/numTestImages;
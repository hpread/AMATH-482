clear; close all; clc

%% Load and data for Test 1: Ideal Case
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
% implay(vidFrames1_1)
% implay(vidFrames2_1)
% implay(vidFrames3_1)

%% Crop video frame
croppedVidFrames1_1 = vidFrames1_1(:,275:400,:,:);
croppedVidFrames2_1 = vidFrames2_1(:,225:375,:,:);
croppedVidFrames3_1 = vidFrames3_1(225:350,:,:,:);
% implay(croppedVidFrames1_1)
% implay(croppedVidFrames2_1)
% implay(croppedVidFrames3_1)

%% Locate position of the can by tracking the most red component in movie
numFrames1 = size(croppedVidFrames1_1,4);
for j = 1:numFrames1
    X1 = croppedVidFrames1_1(:,:,1,j); % row, col, color, time
    [M,I] = max(X1(:));
    [x1,y1] = ind2sub([size(X1,1), size(X1,2)], I);
    posCam1(j,:) = [x1,y1];
end

numFrames2 = size(croppedVidFrames2_1,4);
for j = 1:numFrames2
    X2 = croppedVidFrames2_1(:,:,1,j); % row, col, color, time
    [M,I] = max(X2(:));
    [x2,y2] = ind2sub([size(X2,1), size(X2,2)], I);
    posCam2(j,:) = [x2,y2];
end

numFrames3 = size(croppedVidFrames3_1,4);
for j = 1:numFrames3
    X3 = croppedVidFrames3_1(:,:,1,j); % row, col, color, time
    [M,I] = max(X3(:));
    [x3,y3] = ind2sub([size(X3,1), size(X3,2)], I);
    posCam3(j,:) = [x3,y3];
end

%% Set the mean of the results equal to zero
posCam1x = posCam1(:,2) - mean(posCam1(:,2));
posCam1y = posCam1(:,1) - mean(posCam1(:,1));
posCam2x = posCam2(:,2) - mean(posCam2(:,2));
posCam2y = posCam2(:,1) - mean(posCam2(:,1));
posCam3x = posCam3(:,2) - mean(posCam3(:,2));
posCam3y = posCam3(:,1) - mean(posCam3(:,1));

%% Plot results
subplot(1,3,1)
plot(posCam1x)
hold on
plot(posCam1y)
title('Camera 1')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,2)
plot(posCam2x)
hold on
plot(posCam2y)
title('Camera 2')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,3)
plot(posCam3y)
hold on
plot(posCam3x)
title('Camera 3')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

sgtitle('Position Data for Test 1: Ideal Case')

%% Trim data to line up peaks
tPosCam1x = posCam1x(2:end);
tPosCam1y = posCam1y(2:end);

tPosCam2x = posCam2x(11:length(tPosCam1x)+10);
tPosCam2y = posCam2y(11:length(tPosCam1x)+10);

tPosCam3x = posCam3x(1:length(tPosCam1x));
tPosCam3y = posCam3y(1:length(tPosCam1x));

%% Perform PCA and calculate energies
test1Mat = [tPosCam1x tPosCam1y tPosCam2x tPosCam2y tPosCam3x tPosCam3y];
test1Mat = test1Mat'; % let data vectors be rows of test1Mat
[U1,S1,V1] = svd(test1Mat,'econ');
sig1 = diag(S1);
energies1 = sig1.^2/sum(sig1.^2); % will be plotted later
    
%% Repeat for Test 2: Noisy Case

%% Load data for Test 2: Noisy Case
load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')

%% Crop video frame
croppedVidFrames1_2 = vidFrames1_2(:,275:500,:,:);
croppedVidFrames2_2 = vidFrames2_2(:,175:450,:,:);
croppedVidFrames3_2 = vidFrames3_2(175:350,:,:,:);

%% Locate position of the can by tracking the most red component in movie
numFrames1 = size(croppedVidFrames1_2,4);
for j = 1:numFrames1
    X1 = croppedVidFrames1_2(:,:,1,j); % row, col, color, time
    [M,I] = max(X1(:));
    [x1,y1] = ind2sub([size(X1,1), size(X1,2)], I);
    posCam1(j,:) = [x1,y1];
end

numFrames2 = size(croppedVidFrames2_2,4);
for j = 1:numFrames2
    X2 = croppedVidFrames2_2(:,:,1,j); % row, col, color, time
    [M,I] = max(X2(:));
    [x2,y2] = ind2sub([size(X2,1), size(X2,2)], I);
    posCam2(j,:) = [x2,y2];
end

numFrames3 = size(croppedVidFrames3_2,4);
for j = 1:numFrames3
    X3 = croppedVidFrames3_2(:,:,1,j); % row, col, color, time
    [M,I] = max(X3(:));
    [x3,y3] = ind2sub([size(X3,1), size(X3,2)], I);
    posCam3(j,:) = [x3,y3];
end

%% Set the mean of the results equal to zero
posCam1x = posCam1(:,2) - mean(posCam1(:,2));
posCam1y = posCam1(:,1) - mean(posCam1(:,1));
posCam2x = posCam2(:,2) - mean(posCam2(:,2));
posCam2y = posCam2(:,1) - mean(posCam2(:,1));
posCam3x = posCam3(:,2) - mean(posCam3(:,2));
posCam3y = posCam3(:,1) - mean(posCam3(:,1));

%% Plot results
figure
subplot(1,3,1)
plot(posCam1x)
hold on
plot(posCam1y)
title('Camera 1')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,2)
plot(posCam2x)
hold on
plot(posCam2y)
title('Camera 2')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,3)
plot(posCam3y)
hold on
plot(posCam3x)
title('Camera 3')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

sgtitle('Position Data for Test 2: Noisy Case')

%% Trim data
tPosCam1x = posCam1x(1:end);
tPosCam1y = posCam1y(1:end);

tPosCam2x = posCam2x(26:length(tPosCam1x)+25);
tPosCam2y = posCam2y(26:length(tPosCam1x)+25);

tPosCam3x = posCam3x(7:length(tPosCam1x)+6);
tPosCam3y = posCam3y(7:length(tPosCam1x)+6);

%% Perform PCA and calculate energies
test2Mat = [tPosCam1x tPosCam1y tPosCam2x tPosCam2y tPosCam3x tPosCam3y];
test2Mat = test2Mat'; % let data vectors be rows of test2Mat
[U2,S2,V2] = svd(test2Mat,'econ');
sig2 = diag(S2);
energies2 = sig2.^2/sum(sig2.^2);

%% Repeat for Test 3: Horizontal Displacement

%% Load data for Test 3: Horizontal Displacement
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

%% Crop video frame
croppedVidFrames1_3 = vidFrames1_3(:,275:400,:,:);
croppedVidFrames2_3 = vidFrames2_3(:,175:450,:,:);
croppedVidFrames3_3 = vidFrames3_3(150:350,:,:,:);

%% Locate position of the can by tracking the most red component in movie
numFrames1 = size(croppedVidFrames1_3,4);
for j = 1:numFrames1
    X1 = croppedVidFrames1_3(:,:,1,j); % row, col, color, time
    [M,I] = max(X1(:));
    [x1,y1] = ind2sub([size(X1,1), size(X1,2)], I);
    posCam1(j,:) = [x1,y1];
end

numFrames2 = size(croppedVidFrames2_3,4);
for j = 1:numFrames2
    X2 = croppedVidFrames2_3(:,:,1,j); % row, col, color, time
    [M,I] = max(X2(:));
    [x2,y2] = ind2sub([size(X2,1), size(X2,2)], I);
    posCam2(j,:) = [x2,y2];
end

numFrames3 = size(croppedVidFrames3_3,4);
for j = 1:numFrames3
    X3 = croppedVidFrames3_3(:,:,1,j); % row, col, color, time
    [M,I] = max(X3(:));
    [x3,y3] = ind2sub([size(X3,1), size(X3,2)], I);
    posCam3(j,:) = [x3,y3];
end

%% Set the mean of the results equal to zero
posCam1x = posCam1(:,2) - mean(posCam1(:,2));
posCam1y = posCam1(:,1) - mean(posCam1(:,1));
posCam2x = posCam2(:,2) - mean(posCam2(:,2));
posCam2y = posCam2(:,1) - mean(posCam2(:,1));
posCam3x = posCam3(:,2) - mean(posCam3(:,2));
posCam3y = posCam3(:,1) - mean(posCam3(:,1));

%% Plot results
figure
subplot(1,3,1)
plot(posCam1x)
hold on
plot(posCam1y)
title('Camera 1')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,2)
plot(posCam2x)
hold on
plot(posCam2y)
title('Camera 2')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,3)
plot(posCam3y)
hold on
plot(posCam3x)
title('Camera 3')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

sgtitle('Position Data for Test 3: Horizontal Displacement')

%% Trim data
tPosCam1x = posCam1x(4:end);
tPosCam1y = posCam1y(4:end);

tPosCam2x = posCam2x(24:length(tPosCam1x)+23);
tPosCam2y = posCam2y(24:length(tPosCam1x)+23);

tPosCam3x = posCam3x(1:length(tPosCam1x));
tPosCam3y = posCam3y(1:length(tPosCam1x));

%% Perform PCA and calculate energies
test3Mat = [tPosCam1x tPosCam1y tPosCam2x tPosCam2y tPosCam3x tPosCam3y];
test3Mat = test3Mat'; % let data vectors be rows of test3Mat
[U3,S3,V3] = svd(test3Mat,'econ');
sig3 = diag(S3);
energies3 = sig3.^2/sum(sig3.^2);

%% Repeat for Test 4: Horizontal Displacement and Rotation

%% Load data for Test 4: Horizontal Displacement and Rotation
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')

%% Crop video frame
croppedVidFrames1_4 = vidFrames1_4(:,300:475,:,:);
croppedVidFrames2_4 = vidFrames2_4(:,175:450,:,:);
croppedVidFrames3_4 = vidFrames3_4(125:350,:,:,:);

%% Locate position of the can by tracking the most red component in movie
numFrames1 = size(croppedVidFrames1_4,4);
for j = 1:numFrames1
    X1 = croppedVidFrames1_4(:,:,1,j); % row, col, color, time
    [M,I] = max(X1(:));
    [x1,y1] = ind2sub([size(X1,1), size(X1,2)], I);
    posCam1(j,:) = [x1,y1];
end

numFrames2 = size(croppedVidFrames2_4,4);
for j = 1:numFrames2
    X2 = croppedVidFrames2_4(:,:,1,j); % row, col, color, time
    [M,I] = max(X2(:));
    [x2,y2] = ind2sub([size(X2,1), size(X2,2)], I);
    posCam2(j,:) = [x2,y2];
end

numFrames3 = size(croppedVidFrames3_4,4);
for j = 1:numFrames3
    X3 = croppedVidFrames3_4(:,:,1,j); % row, col, color, time
    [M,I] = max(X3(:));
    [x3,y3] = ind2sub([size(X3,1), size(X3,2)], I);
    posCam3(j,:) = [x3,y3];
end

%% Set the mean of the results equal to zero
posCam1x = posCam1(:,2) - mean(posCam1(:,2));
posCam1y = posCam1(:,1) - mean(posCam1(:,1));
posCam2x = posCam2(:,2) - mean(posCam2(:,2));
posCam2y = posCam2(:,1) - mean(posCam2(:,1));
posCam3x = posCam3(:,2) - mean(posCam3(:,2));
posCam3y = posCam3(:,1) - mean(posCam3(:,1));

%% Plot results
figure
subplot(1,3,1)
plot(posCam1x)
hold on
plot(posCam1y)
title('Camera 1')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,2)
plot(posCam2x)
hold on
plot(posCam2y)
title('Camera 2')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

subplot(1,3,3)
plot(posCam3y)
hold on
plot(posCam3x)
title('Camera 3')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('Horizontal Movement','Vertical Movement','Location','southoutside')

sgtitle('Position Data for Test 4: Horizontal Displacement and Rotation')

%% Trim data
tPosCam1x = posCam1x(3:end);
tPosCam1y = posCam1y(3:end);

tPosCam2x = posCam2x(1:length(tPosCam1x));
tPosCam2y = posCam2y(1:length(tPosCam1x));

tPosCam3x = posCam3x(1:length(tPosCam1x));
tPosCam3y = posCam3y(1:length(tPosCam1x));

%% Perform PCA and calculate energies
test4Mat = [tPosCam1x tPosCam1y tPosCam2x tPosCam2y tPosCam3x tPosCam3y];
test4Mat = test4Mat'; % let data vectors be rows of test4Mat
[U4,S4,V4] = svd(test4Mat,'econ');
sig4 = diag(S4);
energies4 = sig4.^2/sum(sig4.^2);

%% Calculate number of sigmas to reach at least 95% energy
allEnergies = [energies1 energies2 energies3 energies4];
totalEnergies = zeros(4,1); % one for each test
numSigmas = zeros(4,1);
for j = 1:length(numSigmas) % for each test
    for k = 1:length(energies1) % all energy vectors are same length
        curEnergy = totalEnergies(j) + allEnergies(k,j);
        if curEnergy < 0.95
            totalEnergies(j) = curEnergy;
            numSigmas(j) = numSigmas(j) + 1;
        else
            totalEnergies(j) = curEnergy;
            numSigmas(j) = numSigmas(j) + 1;
            break
        end
    end
end

totalEnergies
numSigmas

%% Create plot of columns of V and sigma energies
figure
subplot(4,2,1)
plot(V1(:,2),'k','LineWidth',2)
hold on
plot(V1(:,1),'b','LineWidth',2)
title('Test 1 Components for 95% Energy')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('v_2','v_1','Location','southeast','Orientation','horizontal')

subplot(4,2,2)
plot(energies1,'ko','Linewidth',2)
ylim([0 1])
title('Sigma Energies of Test 1')
xlabel('Singular Values')
ylabel('Fraction of Total Energy')

subplot(4,2,3)
plot(V2(:,4),'y','LineWidth',2)
hold on
plot(V2(:,3),'g','LineWidth',2)
plot(V2(:,2),'k','LineWidth',2)
plot(V2(:,1),'b','LineWidth',2)
title('Test 2 Components for 95% Energy')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('v_4','v_3','v_2','v_1','Location','southeast','Orientation','horizontal')

subplot(4,2,4)
plot(energies2,'ko','Linewidth',2)
ylim([0 1])
title('Sigma Energies of Test 2')
xlabel('Singular Values')
ylabel('Fraction of Total Energy')

subplot(4,2,5)
plot(V3(:,4),'y','LineWidth',2)
hold on
plot(V3(:,3),'g','LineWidth',2)
plot(V3(:,2),'k','LineWidth',2)
plot(V3(:,1),'b','LineWidth',2)
title('Test 3 Components for 95% Energy')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('v_4','v_3','v_2','v_1','Location','southeast','Orientation','horizontal')

subplot(4,2,6)
plot(energies3,'ko','Linewidth',2)
ylim([0 1])
title('Sigma Energies of Test 3')
xlabel('Singular Values')
ylabel('Fraction of Total Energy')

subplot(4,2,7)
plot(V4(:,3),'g','LineWidth',2)
hold on
plot(V4(:,2),'k','LineWidth',2)
plot(V4(:,1),'b','LineWidth',2)
title('Test 4 Components for 95% Energy')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('v_3','v_2','v_1','Location','southeast','Orientation','horizontal')

subplot(4,2,8)
plot(energies4,'ko','Linewidth',2)
ylim([0 1])
title('Sigma Energies of Test 4')
xlabel('Singular Values')
ylabel('Fraction of Total Energy')

%% Plot resulting projections of data against time
figure
U12 = U1';
proj1 = U12(1:2,:)*test1Mat;
subplot(1,4,1)
plot(proj1(2,:),'k','LineWidth',2)
hold on
plot(proj1(1,:),'b','LineWidth',2)
title('Test 1')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('m_2','m_1','Location','southoutside','Orientation','horizontal')

U22 = U2';
proj2 = U22(1:4,:)*test2Mat;
subplot(1,4,2)
plot(proj2(4,:),'y','LineWidth',2)
hold on
plot(proj2(3,:),'g','LineWidth',2)
plot(proj2(2,:),'k','LineWidth',2)
plot(proj2(1,:),'b','LineWidth',2)
title('Test 2')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('m_4','m_3','m_2','m_1','Location','southoutside','Orientation','horizontal')

U32 = U3';
proj3 = U32(1:4,:)*test3Mat;
subplot(1,4,3)
plot(proj3(4,:),'y','LineWidth',2)
hold on
plot(proj3(3,:),'g','LineWidth',2)
plot(proj3(2,:),'k','LineWidth',2)
plot(proj3(1,:),'b','LineWidth',2)
title('Test 3')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('m_4','m_3','m_2','m_1','Location','southoutside','Orientation','horizontal')

U42 = U4';
proj4 = U42(1:3,:)*test4Mat;
subplot(1,4,4)
plot(proj4(3,:),'g','LineWidth',2)
hold on
plot(proj4(2,:),'k','LineWidth',2)
plot(proj4(1,:),'b','LineWidth',2)
title('Test 4')
xlabel('Time (Video Frame Number)')
ylabel('Position')
legend('m_3','m_2','m_1','Location','southoutside','Orientation','horizontal')

sgtitle('Projections of Position Data')
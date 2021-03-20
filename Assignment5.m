clear; close all; clc

%% Load data
vMC = VideoReader('monte_carlo_low.mp4');
vS = VideoReader('ski_drop_low.mp4');

videoMC = read(vMC); 
videoS = read(vS);
% video is height by width by color by time/numFrames

%% Create data matrices
[hMC,wMC,numFramesMC] = size(videoMC,1,2,4);
dataMatMC = zeros(hMC*wMC,numFramesMC);
for j = 1:numFramesMC
    frame = double(rgb2gray(videoMC(:,:,:,j))); % format to double
    frame = reshape(frame,[],1); % reshape to column vector
    dataMatMC(:,j) = frame;
end

[hS,wS,numFramesS] = size(videoS,1,2,4);
dataMatS = zeros(hS*wS,numFramesS);
for j = 1:numFramesS
    frame = double(rgb2gray(videoS(:,:,:,j))); % format to double
    frame = reshape(frame,[],1); % reshape to column vector
    dataMatS(:,j) = frame;
end

% Define time vectors
dt = 1;
tMC = 1:dt:numFramesMC;
tS = 1:dt:numFramesS;

%% Create DMD matrices for Monte Carlo video and compute SVD of X1
XMC = dataMatMC;
X1MC = XMC(:,1:end-1);
X2MC = XMC(:,2:end);
[UMC, SigmaMC, VMC] = svd(X1MC,'econ');

%% Create plot of sigma values
plot(diag(SigmaMC),'ko','Linewidth',2)
title('Sigma Values for Monte Carlo')
xlabel('Singular Values')
ylabel('Sigma Values')

%% Calculate S tilde matrix
rankMC = 25; % visually deduced
UMCr = UMC(:,1:rankMC);
VMCr = VMC(:,1:rankMC);
SigmaMCr = SigmaMC(:,1:rankMC);
SMC = UMCr'*X2MC*VMCr*diag(1./diag(SigmaMCr));

%% Calculate eigenvalues and eigenvectors
[eVMC, DMC] = eig(SMC);
muMC = diag(DMC);
omegaMC = log(muMC)/dt;
phiMC = UMCr*eVMC;

%% Create plot of omega values
figure
plot(real(omegaMC),imag(omegaMC),'o')
hold on
plot([0 0], [-0.5 0.5], 'k')
plot([-0.5 0.1], [0 0], 'k')
title('Omega Values for Monte Carlo')
xlabel('Real Axis')
ylabel('Imaginary Axis')

%% Separate foreground and background omega values
thresh = 0.001;
idxOmMC = find(abs(omegaMC) < thresh); % find omegas close to zero in abs val
% Create subset of omega and phi vectors
omegaSubMC = omegaMC(idxOmMC);
phiSubMC = phiMC(idxOmMC);

%% Create DMD Solution
% Find background DMD Solution
y0MC = phiSubMC\X1MC(:,1); % pseudoinverse to obtain initial conditions

uModesMC = zeros(length(y0MC),numFramesMC);
for k = 1:numFramesMC
   uModesMC(:,k) = y0MC.*exp(omegaSubMC*tMC(k)); 
end
uDMDbackMC = phiSubMC*uModesMC;

% Find foreground DMD solution
uDMDforeMC = XMC-abs(uDMDbackMC);
idxResMC = find(uDMDforeMC < 0);
resMC = zeros(size(uDMDforeMC));
resMC(idxResMC) = uDMDforeMC(idxResMC);
uDMDnewBackMC = resMC+abs(uDMDbackMC);
uDMDnewForeMC = uDMDforeMC-resMC;

% Create full DMD solution
uDMDMC = uDMDnewBackMC + uDMDnewForeMC;

%% Create video of results for Monte Carlo
% figure
% for j = 1:numFramesMC
%     reshaped = reshape(uDMDMC(:,j), [hMC,wMC]);
%     imshow(uint8(reshaped))
%     drawnow
% end

%% Create figure comparing frames from original video and DMD results
figure

subplot(4,4,1)
reshaped = reshape(dataMatMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 Original')

subplot(4,4,2)
reshaped = reshape(dataMatMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 Original')

subplot(4,4,3)
reshaped = reshape(dataMatMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 Original')

subplot(4,4,4)
reshaped = reshape(dataMatMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame Original')

subplot(4,4,5)
reshaped = reshape(uDMDMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 DMD Solution')

subplot(4,4,6)
reshaped = reshape(uDMDMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 DMD Solution')

subplot(4,4,7)
reshaped = reshape(uDMDMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 DMD Solution')

subplot(4,4,8)
reshaped = reshape(uDMDMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame DMD Solution')

subplot(4,4,9)
reshaped = reshape(uDMDnewBackMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 Background')

subplot(4,4,10)
reshaped = reshape(uDMDnewBackMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 Background')

subplot(4,4,11)
reshaped = reshape(uDMDnewBackMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 Background')

subplot(4,4,12)
reshaped = reshape(uDMDnewBackMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame Background')

subplot(4,4,13)
reshaped = reshape(uDMDnewForeMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 Foreground')

subplot(4,4,14)
reshaped = reshape(uDMDnewForeMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 Foreground')

subplot(4,4,15)
reshaped = reshape(uDMDnewForeMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 Foreground')

subplot(4,4,16)
reshaped = reshape(uDMDnewForeMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame Foreground')

sgtitle('Original Frames (in Grayscale) versus DMD Solutions for Monte Carlo')

%% Create figure comparing frames from original video and DMD results
figure

subplot(2,4,1)
reshaped = reshape(dataMatMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 Original')

subplot(2,4,2)
reshaped = reshape(dataMatMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 Original')

subplot(2,4,3)
reshaped = reshape(dataMatMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 Original')

subplot(2,4,4)
reshaped = reshape(dataMatMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame Original')

subplot(2,4,5)
reshaped = reshape(uDMDMC(:,1), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 1 DMD Solution')

subplot(2,4,6)
reshaped = reshape(uDMDMC(:,100), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 100 DMD Solution')

subplot(2,4,7)
reshaped = reshape(uDMDMC(:,200), [hMC,wMC]);
imshow(uint8(reshaped))
title('Frame 200 DMD Solution')

subplot(2,4,8)
reshaped = reshape(uDMDMC(:,numFramesMC), [hMC,wMC]);
imshow(uint8(reshaped))
title('Final Frame DMD Solution')

sgtitle('Original Frames (in Grayscale) versus DMD Solutions for Monte Carlo')

%% Create DMD matrices for ski drop video and compute SVD of X1
XS = dataMatS;
X1S = XS(:,1:end-1);
X2S = XS(:,2:end);
[US, SigmaS, VS] = svd(X1S,'econ');

%% Create plot of sigma values
figure
plot(diag(SigmaS),'ko','Linewidth',2)
title('Sigma Values for Ski Drop')
xlabel('Singular Values')
ylabel('Sigma Values')

%% Create zoomed-in plot of sigma values
figure
plot(diag(SigmaS),'ko','Linewidth',2)
title('Zoomed In Sigma Values for Ski Drop')
xlabel('Singular Values')
ylabel('Sigma Values')
ylim([2000 10000])

%% Calculate S tilde matrix
rankS = 20; % visually deduced
USr = US(:,1:rankS);
VSr = VS(:,1:rankS);
SigmaSr = SigmaS(:,1:rankS);
SS = USr'*X2S*VSr*diag(1./diag(SigmaSr));

%% Calculate eigenvalues and eigenvectors
[eVS, DS] = eig(SS);
muS = diag(DS);
omegaS = log(muS)/dt;
phiS = USr*eVS;

%% Create plot of omega values
figure
plot(real(omegaS),imag(omegaS),'o')
hold on
plot([0 0], [-0.1 0.1], 'k')
plot([-0.5 0.1], [0 0], 'k')
title('Omega Values for Ski Drop')
xlabel('Real Axis')
ylabel('Imaginary Axis')

%% Separate foreground and background omega values
thresh = 0.001;
idxOmS = find(abs(omegaS) < thresh); % find omegas close to zero in abs val
% Create subset of omega and phi vectors
omegaSubS = omegaS(idxOmS);
phiSubS = phiS(idxOmS);

%% Create DMD Solution
% Find background DMD Solution
y0S = phiSubS\X1S(:,1); % pseudoinverse to obtain initial conditions

uModesS = zeros(length(y0S),numFramesS);
for k = 1:numFramesS
   uModesS(:,k) = y0S.*exp(omegaSubS*tS(k)); 
end
uDMDbackS = phiSubS*uModesS;

% Find foreground DMD solution
uDMDforeS = XS-abs(uDMDbackS);
idxResS = find(uDMDforeS < 0);
resS = zeros(size(uDMDforeS));
resS(idxResS) = uDMDforeS(idxResS);
uDMDnewBackS = resS+abs(uDMDbackS);
uDMDnewForeS = uDMDforeS-resS;

% Create full DMD solution
uDMDS = uDMDnewBackS + uDMDnewForeS;

%% Create video of results for ski drop
% figure
% for j = 1:numFramesS
%     reshaped = reshape(uDMDS(:,j), [hS,wS]);
%     imshow(uint8(reshaped))
%     drawnow
% end

%% Create figure comparing frames from original video and DMD results
figure

subplot(4,4,1)
reshaped = reshape(dataMatS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 Original')

subplot(4,4,2)
reshaped = reshape(dataMatS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 Original')

subplot(4,4,3)
reshaped = reshape(dataMatS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 Original')

subplot(4,4,4)
reshaped = reshape(dataMatS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame Original')

subplot(4,4,5)
reshaped = reshape(uDMDS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 DMD Solution')

subplot(4,4,6)
reshaped = reshape(uDMDS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 DMD Solution')

subplot(4,4,7)
reshaped = reshape(uDMDS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 DMD Solution')

subplot(4,4,8)
reshaped = reshape(uDMDS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame DMD Solution')

subplot(4,4,9)
reshaped = reshape(uDMDnewBackS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 Background')

subplot(4,4,10)
reshaped = reshape(uDMDnewBackS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 Background')

subplot(4,4,11)
reshaped = reshape(uDMDnewBackS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 Background')

subplot(4,4,12)
reshaped = reshape(uDMDnewBackS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame Background')

subplot(4,4,13)
reshaped = reshape(uDMDnewForeS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 Foreground')

subplot(4,4,14)
reshaped = reshape(uDMDnewForeS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 Foreground')

subplot(4,4,15)
reshaped = reshape(uDMDnewForeS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 Foreground')

subplot(4,4,16)
reshaped = reshape(uDMDnewForeS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame Foreground')

sgtitle('Original Frames (in Grayscale) versus DMD Solutions for Ski Drop')

%% Create figure comparing frames from original video and DMD results
figure

subplot(2,4,1)
reshaped = reshape(dataMatS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 Original')

subplot(2,4,2)
reshaped = reshape(dataMatS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 Original')

subplot(2,4,3)
reshaped = reshape(dataMatS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 Original')

subplot(2,4,4)
reshaped = reshape(dataMatS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame Original')

subplot(2,4,5)
reshaped = reshape(uDMDS(:,1), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 1 DMD Solution')

subplot(2,4,6)
reshaped = reshape(uDMDS(:,200), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 200 DMD Solution')

subplot(2,4,7)
reshaped = reshape(uDMDS(:,370), [hS,wS]);
imshow(uint8(reshaped))
title('Frame 370 DMD Solution')

subplot(2,4,8)
reshaped = reshape(uDMDS(:,numFramesS), [hS,wS]);
imshow(uint8(reshaped))
title('Final Frame DMD Solution')

sgtitle('Original Frames (in Grayscale) versus DMD Solutions for Ski Drop')
clear; close all; clc

% % Plot and gather data about GNR signal
% figure
% [y1, Fs1] = audioread('GNR.m4a');
% gnrTime = length(y1)/Fs1; % record time in seconds, size of domain
% plot((1:length(y1)),y1);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Sweet Child O Mine');
% p8 = audioplayer(y1,Fs1); playblocking(p8);
% 
% % Define necessary variables
% L1 = gnrTime; % spatial domain, length of recording
% n1 = length(y1); % Fourier modes, length of vector
% t21 = linspace(0,L1,n1+1); % time discretization
% t1 = t21(1:n1); % take first n points only 
% t1 = t1';
% k1 = (1/L1)*[0:(n1/2 - 1) -n1/2:-1]; % leave out 2pi to stay in Hertz
% ks1 = fftshift(k1); % frequency components
% 
% % Define Gabor Transform
% tau1 = linspace(0,L1,101); % center of window
% a1 = 1500; % window size
% 
% for j = 1:length(tau1)
%    g1 = exp(-a1*(t1-tau1(j)).^2); % Gaussian window function
%    gnrFiltered = g1.*y1;
%    FgnrFiltered = fft(gnrFiltered);
%    FgnrFilteredSpec(:,j) = fftshift(abs(FgnrFiltered));
% end
% 
% % Restrict domain of displayed values
% idx11 = find(ks1 > 75,1); % first time ks1 value > 75
% idx21 = find(ks1 > 800,1); % first time ks1 value > 800
% ks1lim = ks1(idx11:idx21);
% Fgnrlim = FgnrFilteredSpec(idx11:idx21,:);
% 
% % Create spectrogram
% figure
% pcolor(tau1,ks1lim,Fgnrlim)
% set(gca,'Fontsize',16)
% colormap(hot)
% xlabel('Time in seconds'), ylabel('Frequency in Hertz')
% shading interp
% yticks([277.18 369.99 415.30 554.37 698.46 739.99])
% yticklabels({'D^b=277.18','G^b=399.99','A^b=415.30','D^b=554.37','F=698.46','G^b=739.99'})
% title('Sweet Child O Mine Guitar Part')
% 
% % Plot and gather data about Floyd signal
% figure
[y2, Fs2] = audioread('Floyd.m4a');
floydTime = length(y2)/Fs2; % record time in seconds, size of domain
% plot((1:length(y2)),y2);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Comfortably Numb');
% p9 = audioplayer(y2,Fs2); playblocking(p9);

% Define necessary variables
L2 = floydTime; % spatial domain, length of recording
n2 = length(y2); % Fourier modes, length of vector
t22 = linspace(0,L2,n2+1); % time discretization
t2 = t22(1:n2); % take first n points only
t2 = t2';
k2 = (1/L2)*[0:((n2+1)/2 - 1) (-n2+1)/2:-1]; % leave out 2pi to stay in Hertz
ks2 = fftshift(k2); % frequency components

% % Define Gabor Transform
% tau2 = 0:0.5:L2; % center of window
% a2 = 100; % window size
% 
% for k = 1:length(tau2)
%    g2 = exp(-a2*(t2-tau2(k)).^2); % Gaussian window function
%    floydFiltered = g2.*y2;
%    FfloydFiltered = fft(floydFiltered);
%    FfloydFilteredSpec(:,k) = fftshift(abs(FfloydFiltered));
% end
% 
% % Restrict domain of displayed values
% idx12 = find(ks2 > 50,1); % first time ks1 value > 50
% idx22 = find(ks2 > 150,1); % first time ks1 value > 150
% ks2lim = ks2(idx12:idx22);
% Ffloydlim = FfloydFilteredSpec(idx12:idx22,:);
% 
% % Create spectrogram
% figure
% pcolor(tau2,ks2lim,Ffloydlim)
% set(gca,'Fontsize',16)
% colormap(hot)
% xlabel('Time in seconds'), ylabel('Frequency in Hertz')
% shading interp
% yticks([82.407 92.499 97.999 110 123.47])
% yticklabels({'E=82.407','G^b=92.499','G=97.999','A=110','B=123.47'})
% title('Comfortably Numb Bass Part')
% 
% % Part 2
% 
% tau2 = 0:0.5:L2; % center of window
% a2 = 100; % window size
% 
% % Only keep frequency information for frequencies below 200
% floydF=fft(y2);
% s2 = abs(ks2)<150; % Shannon window function
% floydF(s2) = 0;
% TfloydSFiltered = ifft(floydF);
% 
% % Perform Gabor Transform
% for k = 1:length(tau2)
%    g2 = exp(-a2*(t2-tau2(k)).^2); % Gaussian window function
%    NewFloydFiltered = g2.*TfloydSFiltered;
%    NewFfloydFiltered = fft(NewFloydFiltered);
%    NewFfloydFilteredSpec(:,k) = fftshift(abs(NewFfloydFiltered));
% end
% 
% % Create spectrogram
% figure
% pcolor(tau2,ks2,NewFfloydFilteredSpec)
% set(gca,'ylim',[50 150],'Fontsize',16)
% colormap(hot)
% xlabel('Time in seconds'), ylabel('Frequency in Hertz')
% shading interp
% yticks([82.407 92.499 97.999 110 123.47])
% yticklabels({'E=82.407','G^b=92.499','G=97.999','A=110','B=123.47'})
% title('Comfortably Numb Bass Part')

% Part 3

tau2 = 0:0.5:20; % center of window
a2 = 900; % window size

% Only keep frequency information for frequencies below 200
y2f=fft(y2);
s21 = abs(ks2) < 750; % Shannon window function
y2f(s21) = 0;
TfloydSFiltered = ifft(y2f);

% Perform Gabor Transform
for k = 1:length(tau2)
   g2 = exp(-a2*(t2-tau2(k)).^2); % Gaussian window function
   NewFloydFiltered = g2.*TfloydSFiltered;
   NewFfloydFiltered = fft(NewFloydFiltered);
   NewFfloydFilteredSpec(:,k) = fftshift(abs(NewFfloydFiltered));
end

% Create spectrogram
figure
pcolor(tau2,ks2,NewFfloydFilteredSpec)
set(gca,'ylim',[250 750],'Fontsize',16)
colormap(hot)
xlabel('Time in seconds'), ylabel('Frequency in Hertz')
shading interp
yticks([261.63 293.65 329.63 349.23 392.00 440.00 493.88 523.55 587.33 659.26 698.46])
yticklabels({'C=261.63','D=293.65','E=329.63','F=349.23','G=392','A=440',...
    'B=493.88','C=523.55','D=587.33','E=659.26','F=698.46'})
title('Comfortably Numb Guitar Part')
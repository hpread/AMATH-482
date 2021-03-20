clear all; close all; clc

load subdata.mat % load data file as 262144x49 (space by time) matrix called subdata
timesteps = size(subdata, 2); % number of time increments
L = 10; % spatial domain 
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); % time discretization
x = x2(1:n); y = x; z = x; % take first n points only (periodicity)
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k); % frequency components
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% Average the spectrum
ave = zeros(n,n,n);
for j=1:timesteps
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Unf = fftn(Un);
    ave = ave + Unf;
end
ave = abs(fftshift(ave))/timesteps;

% Plot average signal
M = max(ave,[],'all');
isosurface(Kx,Ky,Kz,abs(ave)/M,0.5)
axis([-10 10 -10 10 -10 10]), grid on, drawnow
xlabel('frequency in x','Rotation',14)
ylabel('frequency in y','Rotation',-26)
zlabel('frequency in z')

% Define filter, centered about (5,-7,2)
coordX = 5; coordY = -7; coordZ = 2; tau = 0.4;
filter = exp(-tau*(Kx - coordX).^2).*exp(-tau*(Ky - coordY).^2).*exp(-tau*(Kz - coordZ).^2);

% Filter data
for j=1:timesteps
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Unf = fftshift(fftn(Un));
    UnFiltered = filter.*Unf;
    Unt = ifftn(UnFiltered);
    [M,I] = max(abs(Unt(:)));
    [x,y,z] = ind2sub([n,n,n], I);
    pos(j,:) = [X(x,y,z),Y(x,y,z),Z(x,y,z)];
end

% Plot filtering results
figure, plot3(pos(:,1),pos(:,2),pos(:,3),'LineWidth',2), grid on
xlabel('frequency in x','Rotation',14)
ylabel('frequency in y','Rotation',-26)
zlabel('frequency in z')
figure, plot(pos(:,1),pos(:,2),'LineWidth',2), grid on
xlabel('frequency in x')
ylabel('frequency in y')
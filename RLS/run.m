
tic
clc
clear
%% values of the parameters
parameters
wjump=0.0189;

vpeak = 200; vreset = -vpeak;
vinf = 200; % represent the infinity, vpeak-vreset=vinf=200 in [DumontErmentrout2017]

N = 1*10^3; % number of cells


%% Euler integration parameters 
dt = 10^(-3);
dt1=10^(-2);
tend =800;
time = 0:dt:tend;
%tend = Tend*k*abs(VR)/C; % Tend: dimensional; tend: dimensionless

%% heterogeneous parameter, Lorentzian distribution
mu = 0.12;                  % centre
hw = 0.02;                  % half width 
%sigma=0.0001;
g_bar=1.2308;
delta_g=0.2;

% random generation
eta = cauchyrnd(mu,hw,N,1);
g=cauchyrnd(g_bar,delta_g,N,1);
gsyn=1.2308;
% or

% deterministic 

% eta = zeros(N,1);
% for j=1:N
%     eta(j) = mu + hw*tan(pi/2*(2*j-N-1)/(N+1));
% end


%% mean-field model
% Izh_mf,               % Euler integration
%Izh_mf_ode451
%Izh_mf_ode45            % ode45, efficient


%% network model 

Izh_network3


%% save data:
save('Izh_mf_network.mat');

%% plot figures

fig_plot  

toc
%% ============= The end ============



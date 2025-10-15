% By Fudong
% Oct 15, 2025
%



clc
clear
close all

%% ==========  Distribution 1 ======================

N=10^4;
mu = 0.2;
hw = 0.02;  

%% deterministic generation of a Lorentzian distribution
% A typo in [Montbrio2015], it should be "tan", not "atan"

eta_deter = zeros(N,1);
n = zeros(N,1);
for j=1:N
    n(j) = (2*j-N-1)/(N+1);
    eta_deter(j) = mu + hw*tan(pi*n(j)/2);
end


figure(1)
histogram(eta_deter,10000,'Normalization','probability'); 
% % 10000 bins, other options: 'Normalization','pdf'

% histogram(eta_deter,10000,'Normalization','probability',DisplayStyle='stairs'); 
% % default: DisplayStyle='bar'

xlim([-0.5,0.5])
% set(gca,'ytick',[]) % not show values on the y-axis
% yticklabels(yticks*100) % show value/%


ylabel('$p$','interpreter','latex','fontsize',14)
xlabel('$\eta$','interpreter','latex','fontsize',14)

%% random generation of a Lorentzian distribution

eta_rnd = cauchyrnd(mu,hw,N,1);

figure(2)
histogram(eta_rnd,20000,'Normalization','probability');
xlim([-0.5,0.5])
set(gca,'ytick',[])
ylabel('$p$','interpreter','latex','fontsize',14)
xlabel('$\eta$','interpreter','latex','fontsize',14)


%% ==========  Distribution 2 ======================

N=10^3;
mu = 1.8;
hw = 1.4;  

%% deterministic generation of a bimodal Lorentzian distribution

n = zeros(N,1);
eta_deter = zeros(N,1);
for j=1:N/2
    n(j) = (2*j-N/2-1)/(N/2+1);
    eta_deter(j) = -mu + hw*tan(pi*n(j)/2);
end

for j=N/2+1:N
    n(j)=(2*j-3*N/2-1)/(N/2+1);
    eta_deter(j) = mu + hw*tan(pi*n(j)/2);
end

figure(3)
h = histogram(eta_deter,800,'Normalization','probability');
xlim([-20,20])
set(gca,'ytick',[])
ylabel('$p$','interpreter','latex','fontsize',14)
xlabel('$\eta$','interpreter','latex','fontsize',14)

%% random generation of a bimodal Lorentzian distribution

eta_rnd1 = cauchyrnd(-mu,hw,N/2,1);
eta_rnd2 = cauchyrnd(mu,hw,N/2,1);
eta_rnd  = [eta_rnd1; eta_rnd2];

figure(4)
histogram(eta_rnd,2000,'Normalization','probability');
xlim([-20,20])
set(gca,'ytick',[])
ylabel('$p$','interpreter','latex','fontsize',14)
xlabel('$\eta$','interpreter','latex','fontsize',14)

%% ==========  The End ====================

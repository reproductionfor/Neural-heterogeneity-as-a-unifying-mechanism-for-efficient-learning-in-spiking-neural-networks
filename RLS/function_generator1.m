fprintf("Starting the simulation of the network responses to extrinsic stimulation ... ")







I=0;

SMAX = sjump*ones(N,N);  
neff = N;                                                    
%
%% ICs of variables 
          % membrane potentials
%

%
                    % recovery variable

%

            % synaptic gating variable (proportion)
       % to bound s in [0,1] because the synaptic 
%               current is g*s(t)*(er-v), not J*s(t) in [Montbrio2015]
%sstore = zeros(1,init_steps); % store the synaptic variable of the first neuron
                             % because of all-to-all connectivity, the
                             % synaptic variable of each neuron is the same
%
%sr=100;
signals=zeros(n_stims,N,cycle_steps/sr);
start = floor(cycle_steps/sr);
n=length(stim_onsets);


for i=1:n
stim=stim_onsets(i);
I=zeros(stim+cycle_steps,1);
I(stim+1:stim+stim_width)=alpha;
I=gaussian_filter(sigma,1,I);
I=sparse(I);
%keyboard
v = y0(1:1000);
w = y0(1001:2000);
s = y0(2001:end);
s = s + (1-s).*(s>1);

%% store the spike time and index of neuron to plot the rasterplot

%
%% Simulation, Euler integration
%
%fired_inf = find(v >= vinf);
len=stim+cycle_steps;

spike=zeros(N,len/sr);
%spike=sparse(spike);
for j = 1:len
    %j
    v_ = v;                % V_ at the time (i-1)*dt, V at the time i*dt
    w_ = w;
    s_ = s;
    %sstore(i) = s(1);
    
    %%
    n_ref = find(v_ >= vreset & v_ <= vpeak); % neurons not in the refractory period
    
    %% 
    fired_inf = find(v_ >= vinf);
    %firings = [firings; (j-1)*dt + 0*fired_inf, fired_inf];
    if mod(j,sr)==0 
       spike(fired_inf,floor(j/sr))=1;
       
    end    
    v(fired_inf) = -v_(fired_inf);
    w(fired_inf) = w_(fired_inf) + wjump;
    
    %% 
   
    %keyboard
    n_fired = find(v_ < vinf); % neurons not fire
    %keyboard;
    rhs = v_(n_fired).*(v_(n_fired) - alpha) - w_(n_fired) + eta(n_fired)...
                                + I(j) + (er - v_(n_fired)).*g(n_fired).*s_(n_fired);
    v(n_fired) = v_(n_fired) + dt*rhs;
    w(n_fired) = w_(n_fired) + dt*(a*(b*v_(n_fired) - w_(n_fired)));
    
    
    s(n_fired) = s_(n_fired) + dt*(-s_(n_fired)/tsyn) + sum(SMAX(n_fired,fired_inf),2)/neff; % row sum 
    s = s + (1-s).*(s>1);  % to bound s in [0,1];
   
    

    
end
spike_1=spike(:,end - start + 1:end);
spike_1=spike_1';
spike = gaussian_filter(sigma, dt1, spike_1);
spike=spike';

signals(i,:,:)=spike;

fprintf('Finished simulation of trial #%d of %d.\n', i, n);
end

fprintf("Starting the analysis of the function generation capacities of the network ...\n")
clear SMAX spike  spike_1 train_signals
idx_train = train_trials(:) + 1;


num_train = length(idx_train);
train_signals = signals(idx_train,:,:);  % 预分配空间



test_signals  = signals(1,:,:);
%
%#train_signals = signals(idx_train,:,:);
%test_signals  = signals(idx_test,:,:);
%train_signals = signals(train_trials+1,:,:);
%signals(train_trials+1,:,:) = []; 
%test_signals = signals(test_trials+1,:,:);

cs=zeros(size(train_signals, 1),N,N);
for i=1:length(train_trials)
    s=squeeze(train_signals(i,:,:));
    cs(i,:,:) = get_c(s,gamma);
end    
s_mean = squeeze(mean(train_signals, 1));
C = squeeze(mean(cs, 1));         
%C_inv = inv(C); 
%w1= C_inv*s_mean;
w = C\s_mean;
w_readout=w*(target_2');
y=(w_readout')*squeeze(test_signals(1,:,:));
error = y - target_2;
MSE = mean(error.^2);   
plot(y);
hold on
plot(target_2)
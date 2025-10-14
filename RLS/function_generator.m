%% all-to-all coupling with the same weights

addpath('C:\Users\86131\Desktop\脉冲神经网络的稀疏性与噪声对异质网络编码性能的影响\代码整理\文章代码\代码复现\function_generation\functions\')
sr=10;
n_stims=1;
n_tests=1;
sigma=10;
gamma=1e-3;

cycle_steps=floor(T/dt);
stim_onsets = linspace(0, T, n_stims + 1);
stim_onsets = stim_onsets(1:end-1);
stim_onsets = floor(stim_onsets / dt);
stim_width = floor(20.0/dt);
test_trials = 0:n_tests:(n_stims - 1);
train_trials = 0:(n_stims-1);
for i = 1:length(test_trials)
    t = test_trials(i);
    idx = find(train_trials == t, 1);  % 找到匹配项的索引
    if ~isempty(idx)
        train_trials(idx) = [];  % 删除该元素
    end
end

steps=cycle_steps/sr;
t = linspace(0, (T*1e-3)/4, steps);
f1 = 6.0;
f2 = 12.0;
%target_2 = sin(2*pi*f1*t).*sin(2*pi*f2*t);
target_2 = sin(2*pi*f1*t);
fprintf("Finding a stable initial network state ...")

init_steps = floor(T1/dt);
I=0;

SMAX = sjump*ones(N,N);  
neff = N;                                                    
%
%% ICs of variables 
v  = zeros(N,1);          % membrane potentials
%
v_mean = zeros(1,init_steps);% evolution of the mean membrane potential
r_mean=zeros(1,T/dt+1);
r_mean_act=zeros(1,floor((init_steps)*dt1));
v_mean_act=zeros(1,floor((init_steps)*dt1));
%
w = b*v;                    % recovery variable
w_mean = zeros(1,init_steps);% 
w_mean_act=zeros(1,floor((init_steps)*dt1));
%

s = zeros(N,1);             % synaptic gating variable (proportion)
s = s + (1-s).*(s>1);       % to bound s in [0,1] because the synaptic 
%               current is g*s(t)*(er-v), not J*s(t) in [Montbrio2015]
sstore = zeros(1,init_steps); % store the synaptic variable of the first neuron
                             % because of all-to-all connectivity, the
                             % synaptic variable of each neuron is the same
%
%% store the spike time and index of neuron to plot the rasterplot
firings = [];
%
%% Simulation, Euler integration
%
%fired_inf = find(v >= vinf);
for i = 1:init_steps
    %i
    v_ = v;                % V_ at the time (i-1)*dt, V at the time i*dt
    w_ = w;
    s_ = s;
    sstore(i) = s(1);
    
    %%
    n_ref = find(v_ >= vreset & v_ <= vpeak); % neurons not in the refractory period
    v_mean(i) = mean(v_(n_ref));  % at the time (i-1)*dt
    w_mean(i) = mean(w_(n_ref));
    
    %% 
    fired_inf = find(v_ >= vinf);
    %firings = [firings; (i-1)*dt + 0*fired_inf, fired_inf];

    v(fired_inf) = -v_(fired_inf);
    w(fired_inf) = w_(fired_inf) + wjump;
    
    %% 
    noises=randn(N,1);
    %keyboard
    n_fired = find(v_ < vinf); % neurons not fire
    %keyboard;
    rhs = v_(n_fired).*(v_(n_fired) - alpha) - w_(n_fired) + eta(n_fired)...
                                + I + (er - v_(n_fired)).*g(n_fired).*s_(n_fired);
    v(n_fired) = v_(n_fired) + dt*rhs;
    w(n_fired) = w_(n_fired) + dt*(a*(b*v_(n_fired) - w_(n_fired)));
    
    
    s(n_fired) = s_(n_fired) + dt*(-s_(n_fired)/tsyn) +sjump*sum(SMAX(n_fired,fired_inf),2)/neff; % row sum 
    s = s + (1-s).*(s>1);  % to bound s in [0,1];
    r_mean(i)=length(fired_inf)/N./dt;
    

    if (mod(i,1/dt1)==0)
        
        v_mean_act(i.*dt1)=mean(v_mean(i-1/dt1+1:i));
        w_mean_act(i.*dt1)=mean(w_mean(i-1/dt1+1:i));
        r_mean_act(i.*dt1)=mean(r_mean(i-1/dt1+1:i));

    end     
end
%keyboard
%
%% calculation of the instantaneous firing rate
%twin = 2*10^(-2);
%fired_time = firings(:,1);
%
%tmax = find(fired_time <= fired_time(end) - twin);
% tmax(1): the first time index when fired_time + twin > fired_time(end)
%fired_num = zeros(tmax(end)+1,1); 
%fired_num(1) = length(find(fired_time < twin));
%
%for i = 1:tmax(end)
 %   fired_num(i+1) = length(find(fired_time >= fired_time(i) & fired_time < twin + fired_time(i)));    
%end
%avg_fired_time = [0;fired_time(1:tmax(end))];
%keyboard;
%R = fired_num/twin/N; % the firing rate



fprintf("Finished simulation to find a stable initial network state.\n")

y0=[v; w; s];




close all
figure(1)

%% size of the figure (width*height), position shown in the screen
fig=gcf; 
fig.Position=[10,10,700,700]; 
                        
                        
%% raster plot,randomly selected neurons to plot
f1a=subplot(4,1,1);
N_rand = 300; 
index_rand = ceil(N*rand(N_rand,1)); 
for j=1:N_rand
    select = find(firings(:,2)==index_rand(j));
    % the firing times of the jth neuron in the ylabel of raster plot, 
    
    row_num = length(select);
    nn = j*ones(row_num,1);
    plot(firings(select,1),nn,'.k','MarkerSize',0.5);
 
    hold on
end
ylabel('Neuron \#','Interpreter','LaTeX')
xlim([0,tend])


%% population firing rate
f1b=subplot(4,1,2);
%tx=avg_fired_time;
time = 0:dt/dt1:tend-dt/dt1;
plot(time,r_mean_act,'b')                      % network
hold on
plot(t,rm,'r','LineWidth',2);       % mean field
plot(t1,rm1,'g','LineWidth',2);

ylabel('$r(t)$','FontSize',14,'Interpreter','LaTeX')
xlim([0,tend])
hold off

%% mean membrane potential
f1c=subplot(4,1,3);
plot(time,v_mean_act,'b')           % network
hold on 
plot(t,vm,'r','LineWidth',2)    % mean field
plot(t1,vm1,'g','LineWidth',2)
ylabel('$\langle v(t) \rangle$','FontSize',14,'Interpreter','LaTeX')
xlabel('Time','Interpreter','LaTeX')
xlim([0,tend])
hold off


%% mean recovery variable

f1d=subplot(4,1,4);
plot(time,w_mean_act,'b')           % network
hold on
plot(t,wm,'r','LineWidth',2)    % mean field model
plot(t1,wm1,'g','LineWidth',2)

% === Amplitudes of PO ========================
long = length(time);
start = round(long*2/3);

up_w = max(w_mean(start:end))*ones(long,1);
lo_w = min(w_mean(start:end))*ones(long,1);

% plot(time,up_w,'m',time,lo_w,'m')
hold off
%===============================================


ylabel('$\langle w(t) \rangle$','FontSize',14,'Interpreter','LaTeX')
xlabel('Time','Interpreter','LaTeX')
xlim([0,tend])


% IC
rm = zeros(1,tend/dt+1);
rm_act=zeros(1,round((tend/dt+1)*dt1));

vm = zeros(1,tend/dt+1);
vm_act=zeros(1,round((tend/dt+1)*dt1));
wm = b*vm;
wm_act=zeros(1,round((tend/dt+1)*dt1));
sm = zeros(1,tend/dt+1);

sm = sm + (1-sm).*(sm>1);    % to bound sm in [0,1];
%
for i = 1:tend/dt
    i
    rm(i+1) = rm(i) + dt*(hw/pi + 2*rm(i)*vm(i) - rm(i)*(gsyn*sm(i) + alpha));
    vm(i+1) = vm(i) + dt*(vm(i)^2 - alpha*vm(i) + gsyn*sm(i)*(er - vm(i))...
                                          - pi^2*rm(i)^2 - wm(i) + mu + I);
    wm(i+1) = wm(i) + dt*(a*(b*vm(i) - wm(i)) + wjump*rm(i));
    sm(i+1) = sm(i) + dt*(-sm(i)/tsyn + sjump*rm(i)); 
    sm = sm + (1-sm).*(sm>1);  % to bound sm in [0,1];
    if (mod(i,1/dt1)==0)
        
        vm_act(i.*dt1)=mean(vm(i-1/dt1+1:i));
        wm_act(i.*dt1)=mean(wm(i-1/dt1+1:i));
        rm_act(i.*dt1)=mean(rm(i-1/dt1+1:i));

    end   
end

t = 0:dt1:tend;


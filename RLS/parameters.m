

% variables: V,W,S
%
% the revovery eq. of W
beta = -1;             % Resonator/Integrator Variable
TW = 200;              % time constant of the adaptation eq.
VR = -65;              % resting membrane potential
%
% the membrane potential eq. of V
C = 250;              % Capacitance
k1 = 2.5;              % spike width factor
VT = VR + 40 - beta/k1; % threshold
Er = 0;               % Reversal Potential 
Gsyn = 200;           % maximal conductanc
Iext = 0;             % extra current
%
% synaptic eq. of S
Sjump = 0.8;
Tsyn = 4;             % synaptic time constant
%
% the resetting rule
Vpeak = 30;
Vreset = -55;  
Wjump = 200;

% variables
% v = 1 + V/abs(VR);
% w = W/(k*VR*VR);
% s = S; 
%
% the revovery eq. of w
a = C/(TW*k1*abs(VR));
b = beta/(k1*abs(VR));
%
% the membrane potential eq. of v
alpha = 1 + VT/abs(VR);
gsyn = Gsyn/(k1*abs(VR));
er = 1+Er/abs(VR);
I = Iext/(k1*VR*VR);
%
% synaptic eq. of s
tsyn = Tsyn*k1*abs(VR)/C;
sjump = Sjump*C/(k1*abs(VR));  
%
% the resetting rule
vpeak  = 1 + Vpeak/abs(VR);
vreset = 1 + Vreset/abs(VR);
wjump = Wjump/(k1*abs(VR*VR));
%

%% ===========  The End ====================================

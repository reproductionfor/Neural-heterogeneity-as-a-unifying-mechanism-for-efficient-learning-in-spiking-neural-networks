
function dy = mf_sys(mu,hw,alpha,g_bar,delta_g,er,a,b,wjump,tsyn,sjump,I,t,y) 
%
% The mean field model (no delay):
% r'      = hw/pi + 2*r*v_mean - r*(g*s + alpha)
% v_mean' = -w_mean + mu + v_mean^2 + g*s*(er - v_mean)...
%                                - alpha*v_mean - pi^2*r^2
% w_mean' = a*(b*v_mean - w_mean) + wjump * r
% s'      = -s/ts + sjump*r
%
% The heterogeneous source: the applied current
%     the Lorentzian distribution: mu (center), hw (half width)
%
%==========================================================
%
% if y(4) > 1
%     y(4)=1;
% end  
% it seems there is no effect on limit of s in [0,1]
% in our case, because mu, hw is small, s always stays in [0,1], but when
% mu=5, hw=1, s will >1.

dy(1) = hw/pi + 2*y(1)*y(2) - y(1)*(g_bar*y(4) + alpha)+delta_g*y(4)*(er-y(2))/pi;
dy(2) = y(2)^2 - alpha*y(2) + g_bar*y(4)*(er-y(2)) - pi^2*y(1)^2 -y(3) + mu + I+delta_g*y(4)*pi*y(1);
dy(3) = a*(b*y(2) - y(3)) + wjump*y(1);
dy(4) = -y(4)/tsyn + sjump*y(1);
%
end
 
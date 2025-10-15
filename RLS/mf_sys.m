
function dy = mf_sys(mu,hw,alpha,g_bar,delta_g,er,a,b,wjump,tsyn,sjump,I,t,y) 
%


dy(1) = hw/pi + 2*y(1)*y(2) - y(1)*(g_bar*y(4) + alpha)+delta_g*y(4)*(er-y(2))/pi;
dy(2) = y(2)^2 - alpha*y(2) + g_bar*y(4)*(er-y(2)) - pi^2*y(1)^2 -y(3) + mu + I+delta_g*y(4)*pi*y(1);
dy(3) = a*(b*y(2) - y(3)) + wjump*y(1);
dy(4) = -y(4)/tsyn + sjump*y(1);
%
end
 

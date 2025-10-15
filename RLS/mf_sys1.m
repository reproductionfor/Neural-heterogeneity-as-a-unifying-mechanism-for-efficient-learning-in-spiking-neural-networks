function dy = mf_sys1(mu,hw,alpha,gsyn,er,a,b,wjump,tsyn,sjump,I,t,y) 



dy(1) = (hw)/pi + 2*y(1)*y(2) - y(1)*(gsyn*y(4) + alpha);
dy(2) = y(2)^2 - alpha*y(2) + gsyn*y(4)*(er-y(2)) - pi^2*y(1)^2 -y(3) + mu + I;
dy(3) = a*(b*y(2) - y(3)) + wjump*y(1);
dy(4) = -y(4)/tsyn + sjump*y(1);
%dy(5) = 2*sigma^2+4*(y(5)*y(2)-pi*y(6)*y(1));
%dy(6) = 4*(pi*y(5)*y(1)+y(6)*y(2));
%


end


% IC
rint = 0;
vint = 0;
wint = b*vint;
sint = 0;
sint = sint + (1-sint).*(sint>1);% to bound sm in [0,1];
pt=0;
qt=0;
%keyboard
[t1,y] = ode45(@(t,y) mf_sys1(mu,hw,alpha,gsyn,er,a,b,wjump,tsyn,sjump,I,t,y)',[0,tend],[rint,vint,wint,sint]');
rm1 = y(:,1);
vm1 = y(:,2);
wm1 = y(:,3);
sm1 = y(:,4);
%qm = y(:,5);
%pm = y(:,6);
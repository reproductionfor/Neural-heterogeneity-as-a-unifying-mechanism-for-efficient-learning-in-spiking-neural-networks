


% IC
rint = 0;
vint = 0;
wint = b*vint;
sint = 0;
sint = sint + (1-sint).*(sint>1);    % to bound sm in [0,1];

[t,y] = ode45(@(t,y) mf_sys(mu,hw,alpha,g_bar,delta_g,er,a,b,wjump,tsyn,sjump,I,t,y)',[0,tend],[rint,vint,wint,sint]');
rm = y(:,1);
vm = y(:,2);
wm = y(:,3);
sm = y(:,4);




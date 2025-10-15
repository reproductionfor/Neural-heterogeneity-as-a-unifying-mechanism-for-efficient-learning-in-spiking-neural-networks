function x = cauchyrnd(mu,hw,varargin)

x = mu + hw.*tan(pi*(rand(varargin{:}) - 0.5));
end

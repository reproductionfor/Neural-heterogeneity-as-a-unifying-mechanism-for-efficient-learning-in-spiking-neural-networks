function x = cauchyrnd(mu,hw,varargin)
% x = cauchyrnd(mu,hw,M,N)
% to generate M*N random variables with Cauchy (/Lorentizan) distribution
% mu: the location parameter/ center
% hw: the scale parameter/ half width
%
% the cumulative distribution function (CDF):
%   F = (x, mu, hw) = 1/pi * arctan[(x-mu)/hw] + 1/2
% then,
%   x = hw * tan[pi*(F - 1/2)] + mu
%
% F varies from 0 to 1. In code, we can replace F with values randomly 
%  sampled from the uniform distribution on (0,1)
% 
% This technique is referred to as inverse transform sampling and is very
% useful for generating random variates from many distributions.
%
% ref. 
% method 1: (seen in many places)
% https://math.stackexchange.com/questions/484395/how-to-generate-a-cauchy-random-variable
%
% method 2:
% without analytical expression of CDF (not verified yet)
% https://www.mathworks.com/matlabcentral/answers/80333-generate-number-from-a-probability-distribution
%
% method 3: generate fixed values (not proper)
% https://www.mathworks.com/help/stats/work-with-the-cauchy-distribution-using-the-t-location-scale-distribution.html
%
x = mu + hw.*tan(pi*(rand(varargin{:}) - 0.5));
end
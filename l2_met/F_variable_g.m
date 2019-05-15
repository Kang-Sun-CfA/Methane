function variable_g = F_variable_g(inp)
% calculate g as a function of location. only altitude is considered as of
% 2019/04/16
g0 = 9.8;
R_earth = 6371*1e3;
variable_g = g0*(R_earth./(R_earth+inp.H)).^2;

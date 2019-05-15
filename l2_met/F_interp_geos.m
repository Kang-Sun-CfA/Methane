function outp = F_interp_geos(geos,inp)
% resample 2d and 3d fields from regularized geos fp fields. written by
% Kang Sun on 2019/04/17
% inp = [];
% inp.interp_lon = [-80 -100];
% inp.interp_lat = [35 45];
% inp.interp_H = [50 600];
% inp.interp_tai93 = [825508000 825508810];
% inp.interp_lon = oco.lon;
% inp.interp_lat = oco.lat;
% inp.interp_tai93 = oco.tai93;
% inp.interp_H = oco.Z;
% inp.extra2d_interp = {'xco2','TQV'};
% inp.extra3d_interp = {'CO2'};

interp_lon = double(inp.interp_lon(:));
interp_lat = double(inp.interp_lat(:));
interp_tai93 = double(inp.interp_tai93(:));
if isfield(inp,'extra2d_interp')
    extra2d_interp = inp.extra2d_interp;
else
    extra2d_interp = {};
end
if isfield(inp,'extra3d_interp')
    extra3d_interp = inp.extra3d_interp;
else
    extra3d_interp = {};
end

outp = [];

if geos.nstep == 1
    warning('There is only one time step in geos! Ignoring time dimension...')
    PS_met = interpn(geos.lon,geos.lat,geos.PS,...
        interp_lon,interp_lat);
    HS_met = interpn(geos.lon,geos.lat,geos.HS,...
        interp_lon,interp_lat);
    T0_met = interpn(geos.lon,geos.lat,geos.T0,...
        interp_lon,interp_lat);
    lapse_rate = interpn(geos.lon,geos.lat,geos.lapse_rate,...
        interp_lon,interp_lat);
else
    % must-have 2d fields to interpolate
    PS_met = interpn(geos.lon,geos.lat,geos.tai93,geos.PS,...
        interp_lon,interp_lat,interp_tai93);
    HS_met = interpn(geos.lon,geos.lat,geos.HS,interp_lon,interp_lat);
    T0_met = interpn(geos.lon,geos.lat,geos.tai93,geos.T0,...
        interp_lon,interp_lat,interp_tai93);
    lapse_rate = interpn(geos.lon,geos.lat,geos.tai93,geos.lapse_rate,...
        interp_lon,interp_lat,interp_tai93);
end
outp.PS_met = PS_met;
outp.HS_met = HS_met;

if isfield(inp,'interp_H')
    % do hypsometric correction
    interp_H = double(inp.interp_H(:));
else
    interp_H = HS_met;
end

% optional 2d fields to interpolate
for field = extra2d_interp
    if geos.nstep == 1
        outp.(char(field)) = interpn(geos.lon,geos.lat,geos.(char(field)),...
            interp_lon,interp_lat);
    else
        outp.(char(field)) = interpn(geos.lon,geos.lat,geos.tai93,geos.(char(field)),...
            interp_lon,interp_lat,interp_tai93);
    end
end

inp_g = [];
inp_g.H = HS_met;
variable_g = F_variable_g(inp_g);
R_d = 287.058;
if isfield(inp,'interp_PS')
    interp_PS = double(inp.interp_PS(:));
else
    interp_PS = PS_met.*(T0_met./(T0_met-lapse_rate.*(interp_H-HS_met))).^(-variable_g/R_d./lapse_rate);
    interp_PS = double(interp_PS(:));
end
if ~isfield(geos,'commonP')
    warning('It appears that 3d data are not regridded to common pressure grid! Skipping 3d interpolation...')
    return
end
inp_PP = [];
inp_PP.PS = interp_PS;
outp_PP = F_PS2PL_ab(inp_PP);
interp4d_PL = outp_PP.PL;
interp4d_lon = repmat(interp_lon,[1,outp_PP.nlayer]);
interp4d_lat = repmat(interp_lat,[1,outp_PP.nlayer]);
interp4d_tai93 = repmat(interp_tai93,[1,outp_PP.nlayer]);
% 1.5 does not equal to 1.5, ironically. have to make a not so decent
% correction
gridP = double(geos.commonP);
gridP(1) = gridP(1)-1e-5;

if geos.nstep == 1
    % must-have 3d fields to interpolate
    outp.T_met = interpn(geos.lon,geos.lat,gridP,geos.T,...
        interp4d_lon,interp4d_lat,interp4d_PL);
    outp.QV_met = interpn(geos.lon,geos.lat,gridP,geos.QV,...
        interp4d_lon,interp4d_lat,interp4d_PL);
    % optional 3d fields to interpolate
    for field = extra3d_interp
        outp.(char(field)) = interpn(geos.lon,geos.lat,gridP,geos.(char(field)),...
            interp4d_lon,interp4d_lat,interp4d_PL);
    end
else
    % must-have 3d fields to interpolate
    outp.T_met = interpn(geos.lon,geos.lat,gridP,geos.tai93,geos.T,...
        interp4d_lon,interp4d_lat,interp4d_PL,interp4d_tai93);
    outp.QV_met = interpn(geos.lon,geos.lat,gridP,geos.tai93,geos.QV,...
        interp4d_lon,interp4d_lat,interp4d_PL,interp4d_tai93);
    % optional 3d fields to interpolate
    for field = extra3d_interp
        outp.(char(field)) = interpn(geos.lon,geos.lat,gridP,geos.tai93,geos.(char(field)),...
            interp4d_lon,interp4d_lat,interp4d_PL,interp4d_tai93);
    end
end
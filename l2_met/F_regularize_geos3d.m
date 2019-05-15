function geos = F_regularize_geos3d(geos,inp)
% remap geos 3d fields to a regular pressure grid. written by Kang Sun on
% 2019/04/16
if isfield(inp,'commonP')
    commonP = inp.commonP(:);
else
maxPSidx = find(geos.PS == max(geos.PS(:)),1,'first');
[maxPSilon,maxPSilat,maxPSistep] = ind2sub(size(geos.PS),maxPSidx);
commonP = squeeze(geos.PL(maxPSilon,maxPSilat,:,maxPSistep));
commonP = [commonP(:);1.1e5];
end
if isfield(inp,'rfields')
    rfields = inp.rfields(:)';
else
    rfields = {'H','T','QV'};
end
nl = length(commonP);
rgeos = [];
rgeos.commonP = commonP;
for field = rfields
    rgeos.(char(field)) = nan(geos.nlon,geos.nlat,nl,geos.nstep,'single');
end
Rd = 287;
g0 = 9.8;
% T has to come after H
H_idx = find(strcmp(rfields,'H'));
T_idx = find(strcmp(rfields,'T'));
if H_idx > T_idx
    rfields{T_idx} = 'H';
    rfields{H_idx} = 'T';
end
% 313 s for 1 step, 1152 x 721 grid, 3.79e-4 s per grid
for istep = 1:geos.nstep
    for ilon = 1:geos.nlon
        for ilat = 1:geos.nlat
            lapserate = squeeze(geos.lapse_rate(ilon,ilat,istep));
            T0 = squeeze(geos.T0(ilon,ilat,istep));
            xdata = squeeze(geos.PL(ilon,ilat,:,istep));
            P0 = xdata(end);
            H0 = squeeze(geos.H(ilon,ilat,end,istep));
            interp_int = commonP <= P0;
            extrap_int = ~interp_int;
            for field = rfields
                ydata = squeeze(geos.(char(field))(ilon,ilat,:,istep));
                tmp = commonP;
                tmp(interp_int) = interp1(xdata,ydata,commonP(interp_int),...
                    'linear','extrap');
                if strcmp(char(field),'H')
                    tmp(extrap_int) = H0+T0/lapserate ...
                        .*(1-(commonP(extrap_int)/P0).^(lapserate*Rd/g0));
                    rgeos.H(ilon,ilat,:,istep) = tmp;
                    tmpH = tmp;
                elseif strcmp(char(field),'T')
                    tmp(extrap_int) = -lapserate*(tmpH(extrap_int)-H0)+T0;
                    rgeos.T(ilon,ilat,:,istep) = tmp;
                else
                    tmp(extrap_int) = interp1(xdata,ydata,commonP(extrap_int),...
                        'nearest','extrap');
                    rgeos.(char(field))(ilon,ilat,:,istep) = tmp;                    
                end
            end           
        end
    end
end
for field = rfields
    geos.(char(field)) = rgeos.(char(field));
end
geos.commonP = double(commonP);
geos.nl = nl;
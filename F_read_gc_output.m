function outp = F_read_gc_output(inp)
% Matlab function to read the netcdf output from gc tool. Originated from
% Xiong Liu's IDL code. Follow the same naming standard

% Rewritten by Kang Sun on 2017/09/02

fn = inp.fn;
ncid = netcdf.open(fn);
[~,nvars,ngatts] = netcdf.inq(ncid);

for i = 0:ngatts-1
    gattname = netcdf.inqAttName(ncid,netcdf.getConstant('NC_GLOBAL'),i);
    globalattribute.(gattname) = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'),gattname);
end

for i = 0:nvars-1
    varname = netcdf.inqVar(ncid,i);
    variable.(varname) = netcdf.getVar(ncid,i);
end
netcdf.close(ncid);

outp = [];
outp.nw = globalattribute.nwavelengths;
outp.nz = globalattribute.nlayers;
outp.ngas = globalattribute.ngas;
outp.wave = variable.Wavelength;
outp.ps = variable.ps;
outp.ts = variable.ts;
outp.zs = variable.zs;
outp.tmid = (outp.ts(1:outp.nz) + outp.ts(2:outp.nz+1))/2.0;
outp.zmid = (outp.zs(1:outp.nz) + outp.zs(2:outp.nz+1))/2.0;
gases0 = globalattribute.gases;
gases = cell(outp.ngas,1);
for i = 1:outp.ngas
    idx = (i-1)*8+3;
    gases{i} = gases0(idx+1:min(idx+4,length(gases0)));
end
outp.gases = gases;
% outp.lon = globalattribute.lon;
% outp.lat = globalattribute.lat;
% outp.mon = globalattribute.month;
% outp.yr = globalattribute.year;
% outp.day = globalattribute.day;
% outp.utc = globalattribute.utc;
% outp.cfrac = globalattribute.cfrac;
% outp.sza = variable.Solarzenithangle;
% outp.vza = variable.Viewingzenithangle;
% outp.aza = variable.Relativeazimuthangle;
outp.aircol = variable.aircol;
outp.surfalb = variable.surfalb;
outp.rad = variable.radiance;
outp.irrad = variable.irradiance;
outp.gascol = variable.gascol;
if isfield(inp,'gasnorm')
    gasnorm = inp.gasnorm;
else
    gasnorm = ones(outp.ngas,1);
end
for i = 1:outp.ngas
    outp.gascol(:, i) = outp.gascol(:, i) / gasnorm(i);
end
if globalattribute.do_AMF_calc
outp.scatweights = variable.scatweights;
outp.amf = squeeze(variable.amf);
end
outp.gas_jac = squeeze(variable.gas_jac);
outp.gastcol = sum(outp.gascol)';
outp.gascol_jac = squeeze(sum(outp.gas_jac,2))...
    ./repmat(outp.rad,[1,outp.ngas])...
    ./transpose(repmat(outp.gastcol,[1,outp.nw]));
dlnI_dC = outp.gas_jac./permute(repmat(outp.gascol,[1,1,outp.nw]),[3,1,2])...
    ./repmat(outp.rad,[1,outp.nz,outp.ngas]);
outp.gas_jac = dlnI_dC;% jacobians in dlnI/dC

outp.t_jac = zeros(outp.nw,1);
do_T_Jacobian = globalattribute.do_T_Jacobian;
if do_T_Jacobian > 0
    teff = sum(outp.tmid.*outp.aircol)/sum(outp.aircol);
    for i = 1:outp.nz
        outp.t_jac = outp.t_jac+variable.t_jac(:,i);
    end
    outp.t_jac = outp.t_jac./outp.rad./teff;
end

do_sfcprs_Jacobian = globalattribute.do_sfcprs_Jacobian;
outp.sfcprs_jac = zeros(outp.nw,1);
if do_sfcprs_Jacobian > 0
    delp = outp.ps(outp.nz)-outp.ps(outp.nz-1);
    outp.sfcprs_jac = variable.sfcprs_jac ./ outp.rad / delp;
end

if isfield(variable,'surfalb_jac')
outp.surfalb_jac = variable.surfalb_jac./ outp.rad ./ outp.surfalb;
else
    outp.surfalb_jac = zeros(outp.nw,1);
end

aod_jac   = zeros(outp.nw,1);
assa_jac  = aod_jac;
cod_jac   = aod_jac;
cssa_jac  = aod_jac;
cfrac_jac = aod_jac;

rad = outp.rad;

aods0 = variable.aods0;
aod0 = sum(aods0);
cods0 = variable.cods0;
cod0 = sum(cods0);

outp.aods0 = aods0;
outp.aod0 = aod0;
outp.cods0 = cods0;
outp.cod0 = cod0;

if isfield(variable,'aods')
aods = variable.aods;
outp.aods = aods;
end
if isfield(variable,'cods')
cods = variable.cods;
outp.cods = cods;
end
if isfield(variable,'assas')
assas = variable.assas;
outp.assas = assas;
end
if isfield(variable,'cssas')    
cssas = variable.cssas;
outp.cssas = cssas;
end

outp.aod_jac = zeros(outp.nw,1);
outp.assa_jac = zeros(outp.nw,1);
outp.cod_jac = zeros(outp.nw,1);
outp.cssa_jac = zeros(outp.nw,1);
outp.cfrac_jac = zeros(outp.nw,1);

do_aod_jacobian  = globalattribute.do_aod_Jacobian;% note the case sensitivity difference between IDL and matlab
do_assa_jacobian = globalattribute.do_assa_Jacobian;
do_cod_jacobian  = globalattribute.do_cod_Jacobian;
do_cssa_jacobian = globalattribute.do_cssa_Jacobian;
do_cfrac_jacobian = globalattribute.do_cfrac_Jacobian;

if do_aod_jacobian > 0 && aod0 > 0
    for i = 1:outp.nz
        if aods0(i) > 0
            aod_jac = aod_jac+variable.aod_jac(:,i);
        end
    end
    aod_jac = aod_jac./rad./aod0;
    outp.aod_jac = aod_jac;
end

waer = zeros(outp.nw,1);
if do_assa_jacobian > 0 && aod0 > 0
    da = find(aods0 > 0);
    for i = 1:outp.nw
        waer(i) = sum(aods(i,da).*assas(i,da))/sum(aods(i,da));
    end
    for i = 1:outp.nz
        if aods0(i) > 0
            assa_jac = assa_jac+variable.assa_jac(:,i);
        end
    end
    assa_jac = assa_jac./rad./waer;
    outp.assa_jac = assa_jac;
end

if do_cod_jacobian > 0 && cod0 > 0
    for i = 1:oup.nz
        if cods0(i) > 0
            cod_jac = cod_jac+variable.cod_jac(:,i);
        end
    end
    cod_jac = cod_jac./rad./cod0;
    outp.cod_jac = cod_jac;
end

wcld = zeros(outp.nw,1);
if do_cssa_jacobian > 0 && cod0 > 0
    da = find(cods0 > 0);
    for i = 1:outp.nw
        wcld(i) = sum(cods(i,da).*cssas(i,da))/sum(cods(i,da));
    end
    for i = 1:oup.nz
        if cods0(i) > 0
            cssa_jac = cssa_jac+variable.cssa_jac(:,i);
        end
    end
    cssa_jac = cssa_jac./rad./wcld;
    outp.cssa_jac = cssa_jac;
end

if do_cfrac_jacobian > 0
    cfrac_jac = variable.cfrac_jac./rad;
    outp.cfrac_jac = cfrac_jac;
end
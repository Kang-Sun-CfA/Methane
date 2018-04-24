function outp = F_read_gc_output(inp)
% Matlab function to read the netcdf output from gc tool. Originated from
% Xiong Liu's IDL code. Follow the same naming standard

% Rewritten by Kang Sun on 2017/09/02
% modified on 2018/03/06 to add airglow jacobian
% updated on 2018/03/10 to degrade super high res airglow spec first.
% otherwise the airglow lines may be missed.
% updated on 2018/04/24 to add the option of retrieving O2 lines/CIA
% together

if isfield(inp,'combine_lines_cia')
    combine_lines_cia = inp.combine_lines_cia;
else
    combine_lines_cia = false;
end

if isfield(inp,'airglowspec_path')
    do_airglow = true;
    agstruct = load(inp.airglowspec_path);
    if isfield(inp,'VZA')
        VZA = inp.VZA;
    else
        warning('You should provide VZA for airglow!')
        VZA = 0;
    end
    agstruct.sa = agstruct.sa/cos(VZA/180*pi);% airglow larger at larger vza
else
    do_airglow = false;
end

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
    if length(gases{i}) < 4
        new = '    ';
        new(1:length(gases{i})) = gases{i};
        gases{i} = new;
    end
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
if do_airglow
    agspec = F_conv_interp_n(agstruct.w1(:),agstruct.sa(:),median(diff(outp.wave))*2,outp.wave(:));
    agspec(isnan(agspec)) = 0;
    agspec(isinf(agspec)) = 0;
    agspec = agspec(:);
    outp.rad = outp.rad+agspec;
end
outp.irrad = variable.irradiance;
outp.gascol = variable.gascol;
gasnorm_all = {'H2O ',1e22;
    'CO2 ',1e21;
    'O3  ',1e16;
    'N2O ',1e18;
    'CO  ',1e18;
    'CH4 ',1e19;
    'O2  ',1e24;
    'O4  ',1e43;
    };
gasnorm = ones(outp.ngas,1);
for i = 1:outp.ngas
    for j = 1:size(gasnorm_all,1)
        if strcmp(gasnorm_all{j,1},outp.gases{i})
            gasnorm(i) = gasnorm_all{j,2};
        end
    end
end
outp.gasnorm = gasnorm;

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

if combine_lines_cia
    o2_idx = find(strcmp(gases,'O2  '));
    o4_idx = find(strcmp(gases,'O4  '));
    outp.o2col_lines_jac = squeeze(outp.gas_jac(:,:,o2_idx));
    outp.o2tcol_lines_jac = squeeze(outp.gascol_jac(:,o2_idx));
    o2col_total_jac = outp.o2col_lines_jac*0;
    o2tcol_total_jac = outp.o2tcol_lines_jac*0;
    for iz = 1:outp.nz
    o2col_total_jac(:,iz) = double(outp.o2col_lines_jac(:,iz))/double(gasnorm(o2_idx)) ...
        +double(squeeze(outp.gas_jac(:,iz,o4_idx)))/double(gasnorm(o4_idx)) ...
        *2*double(outp.gascol(iz,o2_idx))*double(gasnorm(o2_idx)) ...
        /(abs(outp.zs(iz+1)-outp.zs(iz))*1e5);
    
    o2tcol_total_jac = o2tcol_total_jac...
        +outp.o2col_lines_jac(:,iz)*outp.gascol(iz,o2_idx)...
        +double(squeeze(outp.gas_jac(:,iz,o4_idx)))/double(gasnorm(o4_idx)) ...
        *2*(double(outp.gascol(iz,o2_idx))*double(gasnorm(o2_idx))).^2 ...
        /(abs(outp.zs(iz+1)-outp.zs(iz))*1e5);
    end
    o2tcol_total_jac = o2tcol_total_jac/(outp.gastcol(o2_idx)*outp.gasnorm(o2_idx));

    o2col_total_jac = o2col_total_jac*outp.gasnorm(o2_idx);
    o2tcol_total_jac = o2tcol_total_jac*outp.gasnorm(o2_idx);
    outp.o2col_total_jac = o2col_total_jac;
    outp.o2tcol_total_jac = o2tcol_total_jac;
    outp.gas_jac(:,:,o2_idx) = o2col_total_jac;
    outp.gascol_jac(:,o2_idx) = o2tcol_total_jac;
end

if do_airglow
    outp.airglow_jac = agspec./outp.rad;
else
    outp.airglow_jac = 0*outp.rad;
end
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

if isfield(variable,'gas_xsecs')
    outp.gas_xsecs = variable.gas_xsecs;
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
outp.aods_jac = zeros(outp.nw,outp.nz);
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
        outp.aods_jac(:,i) = variable.aod_jac(:,i)./rad./aods(:,i);
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

function s1_low = F_conv_interp_n(w1,s1,fwhm,common_grid)
% This function convolves s1 with a Gaussian fwhm, resample it to
% common_grid

% Made by Kang Sun on 2016/08/02
% Modified by Kang Sun on 2017/09/06 to vectorize s1 input, up to 3
% dimensions

slit = fwhm/1.66511;% half width at 1e

dw0 = median(diff(w1));
ndx = ceil(slit*2.7/dw0);
xx = (0:ndx*2)*dw0-ndx*dw0;
kernel = exp(-(xx/slit).^2);
kernel = kernel/sum(kernel);
size_s1 = size(s1);
if length(size_s1) == 1
    s1_over = conv(s1, kernel, 'same');
    s1_low = interp1(w1,s1_over,common_grid);
elseif length(size_s1) == 2
    s1_low = repmat(common_grid(:),[1,size_s1(2)]);
    for i = 1:size_s1(2)
        s1_over = conv(s1(:,i), kernel, 'same');
        s1_low(:,i) = interp1(w1,s1_over,common_grid);
    end
elseif length(size_s1) == 3
    s1_low = repmat(common_grid(:),[1,size_s1(2),size_s1(3)]);
    for i = 1:size_s1(2)
        for j = 1:size_s1(3)
            s1_over = conv(s1(:,i,j), kernel, 'same');
            s1_low(:,i,j) = interp1(w1,s1_over,common_grid);
        end
    end
else
    error('Can not handle higher dimensions!!!')
end
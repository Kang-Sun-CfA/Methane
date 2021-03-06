function outp = F_read_spv_output(inp)
% Matlab function to read the netcdf output from splat-vlidort. Written by
% Kang Sun on 2018/08/20

% modified by Kang Sun on 2018/08/26 to add airglow to rad first
% updated by Kang Sun on 2018/10/24 to incorporate brdf jacobians

% inp = [];
% inp.fn = '~/runspv/outp/CH4_1660-1670_0.01_50_45_base_GC_upwelling_output.nc';
% inp.nz = 12;
% inp.O2par_path = '~/CH4/O2.par.html';
% inp.VZA = 0;

outp = [];

if isfield(inp,'O2par_path')
    do_airglow = true;
    if isfield(inp,'O21D_col')
        outp.O21D_col = inp.O21D_col;
    else
        warning('O2 1D column assumed as 2e17 molec/m2!')
        outp.O21D_col = 2e17;
    end
    if isfield(inp,'VZA')
        VZA = inp.VZA;
    else
        warning('You should provide VZA for airglow!')
        VZA = 0;
    end
else
    do_airglow = false;
end

if ~isfield(inp,'if_lnR')
    if_lnR = true;
else
    if_lnR = inp.if_lnR;
end

fn = inp.fn;
ncid = netcdf.open(fn);
[~,nvars,ngatts] = netcdf.inq(ncid);
outp.ngas = 0;
gases0 = cell(0);
outp.naerosol = 0;
aerosols0 = cell(0);

for i = 0:nvars-1
    varname = netcdf.inqVar(ncid,i);
    startIndex = regexp(varname,'_gas_jac');
    if ~isempty(startIndex)
        outp.ngas = outp.ngas+1;
        gases0 = cat(1,gases0,varname(1:startIndex-1));
    end
    startIndex = regexp(varname,'_aod_jac');
    if ~isempty(startIndex)
        outp.naerosol = outp.naerosol+1;
        aerosols0 = cat(1,aerosols0,varname(1:startIndex-1));
    end
    variable.(varname) = netcdf.getVar(ncid,i);
end
netcdf.close(ncid);

outp.gases0 = gases0;
outp.aerosols0 = aerosols0;

outp.nw = length(variable.Wavelength);
outp.nz0 = length(variable.zs)-1;
if isfield(inp,'nz')
    outp.nz = inp.nz;
else
    outp.nz = outp.nz0;
end

outp.wave = double(variable.Wavelength);
gases = cell(outp.ngas,1);
for i = 1:outp.ngas
    gases{i} = gases0{i};
    if length(gases{i}) < 4
        new = '    ';
        new(1:length(gases{i})) = gases{i};
        gases{i} = new;
    end
end

outp.gases = gases;

aerosols = cell(outp.naerosol,1);
for i = 1:outp.naerosol
    aerosols{i} = aerosols0{i};
    if length(aerosols{i}) < 4
        new = '    ';
        new(1:length(aerosols{i})) = aerosols{i};
        aerosols{i} = new;
    end
end

outp.aerosols = aerosols;
outp.sza = variable.Solarzenithangle;
outp.vza = variable.Viewingzenithangle;
if isfield(variable,'surfalb')
outp.surfalb = variable.surfalb;
else
    outp.surfalb = variable.BRDF_f_isotr;
    outp.BRDF_f_isotr = variable.BRDF_f_isotr;
end
outp.rad = variable.radiance;
outp.SZA = variable.Solarzenithangle;
outp.VZA = variable.Viewingzenithangle;
% dlnI/dln O2_1D column
if do_airglow
    lines = F_import_par(inp.O2par_path);
    wStart = 1240;
    wEnd = 1300;
    wgrid = linspace(wStart,wEnd,round((wEnd-wStart)/0.0005));
    vStart =  1e7/wEnd;
    vEnd = 1e7/wStart;
    vgrid = 1e7./wgrid;
    hT = 290;
    airglowQ = [180,90.620796;181,91.113693;182,91.606583;183,92.099487;184,92.592407;185,93.085342;186,93.578270;187,94.071220;188,94.564171;189,95.057129;190,95.550102;191,96.043068;192,96.536049;193,97.029030;194,97.522018;195,98.015007;196,98.508011;197,99.000999;198,99.494003;199,99.987000;200,100.48000;201,100.97300;202,101.46600;203,101.95900;204,102.45199;205,102.94499;206,103.43798;207,103.93097;208,104.42395;209,104.91693;210,105.40990;211,105.90287;212,106.39583;213,106.88878;214,107.38173;215,107.87466;216,108.36759;217,108.86051;218,109.35342;219,109.84631;220,110.33920;221,110.83207;222,111.32494;223,111.81778;224,112.31062;225,112.80344;226,113.29624;227,113.78903;228,114.28180;229,114.77456;230,115.26730;231,115.76002;232,116.25272;233,116.74541;234,117.23807;235,117.73071;236,118.22333;237,118.71593;238,119.20851;239,119.70107;240,120.19360;241,120.68611;242,121.17859;243,121.67105;244,122.16348;245,122.65589;246,123.14827;247,123.64062;248,124.13294;249,124.62524;250,125.11750;251,125.60973;252,126.10194;253,126.59411;254,127.08625;255,127.57836;256,128.07043;257,128.56248;258,129.05449;259,129.54646;260,130.03841;261,130.53030;262,131.02217;263,131.51401;264,132.00578;265,132.49754;266,132.98924;267,133.48093;268,133.97256;269,134.46416;270,134.95570;271,135.44720;272,135.93867;273,136.43010;274,136.92148;275,137.41281;276,137.90410;277,138.39536;278,138.88654;279,139.37770;280,139.86880;281,140.35986;282,140.85086;283,141.34183;284,141.83273;285,142.32359;286,142.81439;287,143.30515;288,143.79585;289,144.28650;290,144.77710;291,145.26764;292,145.75813;293,146.24857;294,146.73894;295,147.22926;296,147.71953;297,148.20973;298,148.69987;299,149.18997;300,149.67999];
    hQ = interp1(airglowQ(:,1),airglowQ(:,2),hT,'linear','extrap');
    int = lines.transitionWavenumber > vStart & lines.transitionWavenumber < vEnd;
    
    hg = lines.upperStatisticalWeight(int);
    hE = lines.lowerStateEnergy(int);
    nu00 = 7883.756645;
    hnu = lines.transitionWavenumber(int);
    hA = lines.einsteinACoefficient(int);
    
    MW = 31.98983/1000;
    % speed of light in SI unit
    c = 2.99792458e8;
    % Planck constant in SI unit
    h = 6.62607004e-34;
    % Bolzmann constant in SI unit
    kB = 1.38064852e-23;
    % Avogadro's Constant in SI unit
    Na = 6.02214e23;
    % second radiation constant, 1.4388 cm K, note it is cm
    c2 = h*c/kB*100;
    
    hspike = hA.*hg.*exp(-c2*(hE+hnu-nu00)/hT)/hQ;
    airglow_jac = zeros(size(wgrid),'single');
    F_gauss = @(nu,nu0,hw1e) pi^(-0.5)/hw1e *exp(-((nu-nu0).^2)/hw1e^2);
    for i = 1:length(hspike)
        HW1e = (hnu(i)/c).*sqrt(2*kB*Na*hT./MW);
        vfilter = vgrid > hnu(i)-1 & vgrid < hnu(i)+1;
        airglow_jac(vfilter) = airglow_jac(vfilter)+hspike(i)*F_gauss(vgrid(vfilter),hnu(i),HW1e);
        %     trapz(vgrid(vfilter),F_gauss(vgrid(vfilter),hnu(i),HW1e))
    end
    airglow_jac = airglow_jac.*vgrid./wgrid/4/pi/cos(VZA/180*pi);
    agspec = F_conv_interp_n(wgrid(:),airglow_jac(:),median(diff(outp.wave))*2,outp.wave(:));
    agspec(isnan(agspec)) = 0;
    agspec(isinf(agspec)) = 0;
    agspec = agspec(:);
    outp.rad = outp.rad+agspec*outp.O21D_col;
    outp.airglow_jac = agspec./outp.rad*outp.O21D_col;
    outp.agspec = agspec;
else
    outp.airglow_jac = 0*outp.rad;
end

outp.irrad = variable.irradiance;
dividx = round(linspace(1,outp.nz0+1,outp.nz+1));
outp.ps0 = variable.ps;
outp.ts0 = variable.ts;
outp.zs0 = variable.zs;

outp.aircol = zeros(outp.nz,1,'single');
outp.ps = zeros(outp.nz,1,'single');
for iz = 1:outp.nz
    outp.aircol(iz) = sum(variable.aircol(dividx(iz):dividx(iz+1)-1));
    outp.ps(iz) = mean(variable.ps(dividx(iz):dividx(iz+1)-1));
end
outp.ts = interp1(variable.ps,variable.ts,outp.ps,'linear','extrap');
outp.zs = interp1(variable.ps,variable.zs,outp.ps,'linear','extrap');

gasnorm = [];
gasnorm.H2O = 1e22;
gasnorm.CO2 = 1e21;
gasnorm.O3 = 1e16;
gasnorm.N2O = 1e18;
gasnorm.CO = 1e18;
gasnorm.CH4 = 1e19;
gasnorm.O2 = 1e24;
gasnorm.O4 = 1e43;

gasvmrnorm = [];
gasvmrnorm.H2O = 1e-3;
gasvmrnorm.CO2 = 1e-6;
gasvmrnorm.O3 = 1e-9;
gasvmrnorm.N2O = 1e-9;
gasvmrnorm.CO = 1e-9;
gasvmrnorm.CH4 = 1e-9;
gasvmrnorm.O2 = 1e0;
gasvmrnorm.O4 = 1e21;

outp.gasnorm = gasnorm;
outp.gasvmrnorm = gasvmrnorm;

for i = 1:outp.ngas
    outp.([gases0{i},'_gascol']) = zeros(outp.nz,1,'single');
    outp.([gases0{i},'_gas_jac']) = zeros(outp.nw,outp.nz,'single');
    outp.([gases0{i},'_vmr_jac']) = zeros(outp.nw,outp.nz,'single');
    outp.([gases0{i},'_vmr']) = zeros(outp.nz,1,'single');
    outp.([gases0{i},'_vmrscale_jac']) = zeros(outp.nw,1,'single');
    outp.([gases0{i},'_gascol_jac']) = zeros(outp.nw,1,'single');
    for iz = 1:outp.nz
        % VCD
        outp.([gases0{i},'_gascol'])(iz) = sum(variable.([gases0{i},'_gascol'])(dividx(iz):dividx(iz+1)-1));
        % dI/dVCD
        outp.([gases0{i},'_gas_jac'])(:,iz) = ...
            sum(variable.([gases0{i},'_gas_jac'])(:,dividx(iz):dividx(iz+1)-1),2)/outp.([gases0{i},'_gascol'])(iz);
        % VMR
        outp.([gases0{i},'_vmr'])(iz) = outp.([gases0{i},'_gascol'])(iz)/outp.aircol(iz);
        % dI/dVMR
        outp.([gases0{i},'_vmr_jac'])(:,iz)...
            = outp.aircol(iz)*outp.([gases0{i},'_gas_jac'])(:,iz);
        % dI/dVMR_scale_factor
        outp.([gases0{i},'_vmrscale_jac']) = outp.([gases0{i},'_vmrscale_jac'])...
            +outp.([gases0{i},'_vmr_jac'])(:,iz)*outp.([gases0{i},'_vmr'])(iz);
    end
    % dlnI/dVMR_scale_factor
    outp.([gases0{i},'_vmrscale_jac']) = outp.([gases0{i},'_vmrscale_jac'])./outp.rad;
    % dlnI/dln total VCD, the same as dlnI/dVMR_scale_factor
    outp.([gases0{i},'_gastcol']) = sum(variable.([gases0{i},'_gascol']));
    outp.([gases0{i},'_gascol_jac']) = sum(variable.([gases0{i},'_gas_jac']),2)./outp.rad;
    
    % do some urgly repair
    for iz = 1:outp.nz
    % dlnI/d VCD, VCD normalized by gasnorm
    outp.([gases0{i},'_gas_jac'])(:,iz) = outp.([gases0{i},'_gas_jac'])(:,iz)./outp.rad * gasnorm.(gases0{i});
    % dlnI/d VMR, VMR normalized by gasvmrnorm
    outp.([gases0{i},'_vmr_jac'])(:,iz) = outp.([gases0{i},'_vmr_jac'])(:,iz)./outp.rad * gasvmrnorm.(gases0{i});
    % VCD, normalized by gasnorm
    outp.([gases0{i},'_gascol'])(iz) = outp.([gases0{i},'_gascol'])(iz)/gasnorm.(gases0{i});
    % VMR, normalized by gasvmrnorm
    outp.([gases0{i},'_vmr'])(iz) = outp.([gases0{i},'_vmr'])(iz)/gasvmrnorm.(gases0{i});
%     % dlnI/dln VCD
%     % what I divided, I multiply them back...
%     outp.([gases0{i},'_gas_jac'])(:,iz) = outp.([gases0{i},'_gas_jac'])(:,iz) ...
%         * outp.([gases0{i},'_gascol'])(iz)./outp.rad;
%     % dlnI/dln VMR
%     outp.([gases0{i},'_vmr_jac'])(:,iz) = outp.([gases0{i},'_vmr_jac'])(:,iz) ...
%         * outp.([gases0{i},'_vmr'])(iz)./outp.rad;
    end
    % total VCD, normalized by gasnorm
    outp.([gases0{i},'_gastcol']) = outp.([gases0{i},'_gastcol'])/gasnorm.(gases0{i});
    % output gas xsec, added for proxy work on 2019/02/09
    outp.([gases0{i},'_gas_xsec']) = variable.([gases0{i},'_gas_xsec']);
end
% plot(outp.wave,outp.CH4_vmrscale_jac,outp.wave,outp.CH4_gascol_jac*outp.CH4_gastcol,'.')
% dlnI/dT_shift
outp.t_jac = zeros(outp.nw,1,'single');
for i = 1:outp.nz0
    outp.t_jac = outp.t_jac+variable.t_jac(:,i)/variable.ts(i+1);
end
outp.t_jac = outp.t_jac./outp.rad;
% dlnI/dP_surf
outp.sfcprs_jac = variable.sfcprs_jac/variable.ps(end)./outp.rad;

% dlnI/dAOD, dlnI/dpkh, dlnI/dhfw
for i = 1:outp.naerosol
    outp.([aerosols0{i},'_aod_tau']) = variable.([aerosols0{i},'_aod_tau']);
    outp.([aerosols0{i},'_aod_tau_jac']) = variable.([aerosols0{i},'_aod_tau_jac'])/sum(variable.([aerosols0{i},'_aods0']))./outp.rad;
    
    outp.([aerosols0{i},'_aod_pkh']) = variable.([aerosols0{i},'_aod_pkh']);
    outp.([aerosols0{i},'_aod_pkh_jac']) = variable.([aerosols0{i},'_aod_pkh_jac'])/variable.([aerosols0{i},'_aod_pkh'])./outp.rad;
    
    outp.([aerosols0{i},'_aod_hfw']) = variable.([aerosols0{i},'_aod_hfw']);
    outp.([aerosols0{i},'_aod_hfw_jac']) = variable.([aerosols0{i},'_aod_hfw_jac'])/variable.([aerosols0{i},'_aod_hfw'])./outp.rad;
end

% dlnI/dsurfalb
if isfield(variable,'surfalb_jac')
outp.surfalb_jac = variable.surfalb_jac./ variable.surfalb./outp.rad;
else
    outp.surfalb_jac = variable.BRDF_f_isotr_jac./ variable.BRDF_f_isotr./outp.rad;
end
outp.if_lnR = if_lnR;
if ~if_lnR
    outpfn = fieldnames(outp);
    for ifn = 1:length(outpfn)
        if contains(outpfn{ifn},'_jac')
        if strcmp(outpfn{ifn}(end-3:end),'_jac')
            outp.(outpfn{ifn}) = outp.(outpfn{ifn}).*outp.rad;
%             disp([outpfn{ifn},' unnormalized!']);
        end
        end
    end
end
outp.ods = variable.ods;

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
function outp = F_gas_sens_spv(inp,outp_d)
% the spv version of F_gas_sens.m. Written by Kang Sun on 2018/08/24

if isfield(inp,'albsnorm')
    albsnorm = inp.albsnorm;
else
    albsnorm = 1;
end
inp.albs_aperr = inp.albs_aperr*albsnorm;

outp = struct;
all_gases = {};
for iwin = 1:length(outp_d)
    all_gases = cat(1,all_gases,outp_d(iwin).gases(:));
end
all_gases = unique(all_gases);

if isfield(inp,'included_gases')
    included_gases = inp.included_gases;
else
    included_gases = all_gases;
end

for i = 1:length(included_gases)
    new = '    ';
    new(1:length(included_gases{i})) = included_gases{i};
    if ~ismember(new,all_gases)
        error(['The molecule ',included_gases{i},' you asked for was not simulated by GC tool!!!'])
    end
end

if isfield(inp,'if_vmr')
    if_vmr = inp.if_vmr;
else
    if_vmr = true(length(included_gases),1);
end

if isfield(inp,'inc_prof')
    inc_prof = inp.inc_prof;
    nprof = sum(inc_prof);
    if length(inp.gasprof_aperr_scale_LT) < nprof
        error('Please specify prior error for ALL gas profiles!')
    end
else
    inc_prof = zeros(1,length(all_gases));
end

if isfield(inp,'inc_alb')
    inc_alb = inp.inc_alb;
else
    inc_alb = 0;
end
if isfield(inp,'inc_t')
    inc_t = inp.inc_t;
else
    inc_t = 0;
end
if isfield(inp,'inc_aod')
    inc_aod = inp.inc_aod;
else
    inc_aod = 0;
end

if isfield(inp,'inc_aod_pkh')
    inc_aod_pkh = inp.inc_aod_pkh;
else
    inc_aod_pkh = 0;
end
if isfield(inp,'inc_aod_hfw')
    inc_aod_hfw = inp.inc_aod_hfw;
else
    inc_aod_hfw = 0;
end
if isfield(inp,'inc_assa')
    inc_assa = inp.inc_assa;
else
    inc_assa = 0;
end
if isfield(inp,'inc_cod')
    inc_cod = inp.inc_cod;
else
    inc_cod = 0;
end
if isfield(inp,'inc_cssa')
    inc_cssa = inp.inc_cssa;
else
    inc_cssa = 0;
end
if isfield(inp,'inc_cfrac')
    inc_cfrac = inp.inc_cfrac;
else
    inc_cfrac = 0;
end
if isfield(inp,'inc_sfcprs')
    inc_sfcprs = inp.inc_sfcprs;
else
    inc_sfcprs = 0;
end
if isfield(inp,'inc_airglow')
    inc_airglow = inp.inc_airglow;
else
    inc_airglow = 0;
end
ngas = length(included_gases);
% number of aerosols has to be the same across all windows!
aerosols = inp.included_aerosols;
naerosol = length(aerosols);
for ia = 1:naerosol
    if ~ismember(aerosols{ia},outp_d(1).aerosols0)
        error([aerosols{ia},'is not simulated by spv!'])
    end
end
ngas_vmr = double(sum(if_vmr));
gasjac = [];
gasvmrjac = [];
gascoljac = [];
albjac = [];
tjac = [];
aodjac = [];
assajac = [];
aodpkhjac = [];
aodhfwjac = [];
codjac = [];
cssajac = [];
cfracjac = [];
sfcprsjac = [];
airglowjac = [];
wsnr = [];
rad = [];

nz = outp_d(1).nz;
gascol = zeros(nz,ngas);
gasvmr = zeros(nz,ngas);
gastcol = zeros(ngas,1);
aodtau = zeros(naerosol,1);
aodpkh = zeros(naerosol,1);
aodhfw = zeros(naerosol,1);
% number of albedo terms for all windows
nalb_all = 0;
for iwin = 1:length(outp_d)
    nalb_all = nalb_all+outp_d(iwin).nalb;
end
alb_count = 1;
outp.nw = [];

% concatenate jacobians across windows
for iwin = 1:length(outp_d)
    outp.nw = cat(1,outp.nw,outp_d(iwin).nw);
    tmp_col_jac = zeros(outp_d(iwin).nw,ngas);
    tmp_prof_jac = zeros(outp_d(iwin).nw,outp_d(iwin).nz,ngas);
    % align the gas dimension for different windows. Pad with zeros if a
    % certain gas does not exist in a certain window
    for ig = 1:ngas
        for igc = 1:outp_d(iwin).ngas
            if strcmp(included_gases{ig},outp_d(iwin).gases{igc}(1:length(included_gases{ig})))
                gascol(:,ig) = outp_d(iwin).([included_gases{ig},'_gascol']);
                gasvmr(:,ig) = outp_d(iwin).([included_gases{ig},'_vmr']);
                gastcol(ig) = outp_d(iwin).([included_gases{ig},'_gastcol']);
                tmp_col_jac(:,ig) = outp_d(iwin).([included_gases{ig},'_gascol_jac']);
                if inc_prof(ig)
                    if if_vmr(ig)
                        tmp_prof_jac(:,:,ig) = outp_d(iwin).([included_gases{ig},'_vmr_jac']);
                    else
                        tmp_prof_jac(:,:,ig) = outp_d(iwin).([included_gases{ig},'_gas_jac']);
                    end
                end
            end
        end
    end
    gascoljac = cat(1,gascoljac,tmp_col_jac);
    gasjac = cat(1,gasjac,tmp_prof_jac);
    

        tmp_aod_jac = zeros(outp_d(iwin).nw,naerosol);
        for ia = 1:naerosol
            aodtau(ia) = outp_d(iwin).([aerosols{ia},'_aod_tau']);
            tmp_aod_jac(:,ia) = outp_d(iwin).([aerosols{ia},'_aod_tau_jac']);
        end
        aodjac = cat(1,aodjac,tmp_aod_jac);

    

        tmp_aod_pkh_jac = zeros(outp_d(iwin).nw,naerosol);
        for ia = 1:naerosol
            aodpkh(ia) = outp_d(iwin).([aerosols{ia},'_aod_pkh']);
            tmp_aod_pkh_jac(:,ia) = outp_d(iwin).([aerosols{ia},'_aod_pkh_jac']);
        end
        aodpkhjac = cat(1,aodpkhjac,tmp_aod_pkh_jac);

    

        tmp_aod_hfw_jac = zeros(outp_d(iwin).nw,naerosol);
        for ia = 1:naerosol
            aodhfw(ia) = outp_d(iwin).([aerosols{ia},'_aod_hfw']);
            tmp_aod_hfw_jac(:,ia) = outp_d(iwin).([aerosols{ia},'_aod_hfw_jac']);
        end
        aodhfwjac = cat(1,aodhfwjac,tmp_aod_hfw_jac);

    
    wsnr = cat(1,wsnr,outp_d(iwin).wsnr);
    rad = cat(1,rad,outp_d(iwin).rad);
    
    tmp_albjec = zeros(outp_d(iwin).nw,nalb_all);
    tmp_albjec(:,alb_count:alb_count+outp_d(iwin).nalb-1) = outp_d(iwin).surfalb_jac;
    alb_count = alb_count+outp_d(iwin).nalb;

        albjac = cat(1,albjac,tmp_albjec);


        airglowjac = cat(1,airglowjac,outp_d(iwin).airglow_jac);


        tjac = cat(1,tjac,outp_d(iwin).t_jac);


        sfcprsjac = cat(1,sfcprsjac,outp_d(iwin).sfcprs_jac);

    if inc_assa
        assajac = cat(1,assajac,outp_d(iwin).assa_jac);
    end
    if inc_cod
        codjac = cat(1,codjac,outp_d(iwin).cod_jac);
    end
    if inc_cssa
        cssajac = cat(1,cssajac,outp_d(iwin).cssa_jac);
    end
end
albjac = albjac/albsnorm;

outp.rad = rad;
outp.wsnr = wsnr;
outp.gascol = gascol;
outp.gastcol = gastcol;
outp.gasvmr = gasvmr;

% set up state vectors, and a priori error
nv = 1;% note the difference indexing between matlab and idl
aperrs = [];   %no need remove first value later
varnames = cell(0); %no need remove first value later

gasfidxs = zeros(ngas,1);
gasnvar  = zeros(ngas,1);

% construct gases a priori
gascol_aperr = inp.gascol_aperr_scale(1:ngas);%*gastcol;% dlnI/dlnx now, not dlnI/dx any more

gasprof_aperr_scale = zeros(outp_d(1).nz,nprof); 
for iprof = 1:nprof
gasprof_aperr_scale(outp_d(1).zs <= 2,iprof) = inp.gasprof_aperr_scale_LT(iprof);
gasprof_aperr_scale(outp_d(1).zs > 2 & outp_d(1).zs <= 17,iprof) = inp.gasprof_aperr_scale_UT(iprof);
gasprof_aperr_scale(outp_d(1).zs > 17,iprof) = inp.gasprof_aperr_scale_ST(iprof);
end

gasprof_aperr = gascol;
iprof = 0;
for ig = 1:ngas
    if inc_prof(ig)
        iprof = iprof+1;
        % if dlnI/dlnx
%         gasprof_aperr(:,ig) = gasprof_aperr_scale(:,iprof);
        % if dlnI/dx
        if if_vmr(ig)
            gasprof_aperr(:,ig) = gasvmr(:,ig).*gasprof_aperr_scale(:,iprof);
        else
            gasprof_aperr(:,ig) = gascol(:,ig).*gasprof_aperr_scale(:,iprof);
        end
    end
end
% gasprof_aperr = gascol.*repmat(gasprof_aperr_scale,[1,ngas]);
outp.gasprof_aperr = gasprof_aperr;

albs_aperr = inp.albs_aperr; 
t_aperr = inp.t_aperr; 
aod_aperr = inp.aod_aperr;
aod_pkh_aperr = inp.aod_pkh_aperr;
aod_hfw_aperr = inp.aod_hfw_aperr;
assa_aperr = inp.assa_aperr;
cod_aperr = inp.cod_aperr;
cssa_aperr = inp.cssa_aperr;
cfrac_aperr = inp.cfrac_aperr;
sfcprs_aperr = inp.sfcprs_aperr;
airglow_aperr = inp.airglow_aperr;
for ig = 1: ngas
    gasfidxs(ig) = nv;
    
    if inc_prof(ig)
        tempcell = cell(nz,1);
        for icell = 1:nz
            tempcell{icell} = [included_gases{ig},'-',num2str(icell)];
        end
        varnames = cat(1,varnames,tempcell);
        aperrs = cat(1,aperrs,gasprof_aperr(:,ig));
        gasnvar(ig) = nz;
        nv = nv + nz;
    else
        varnames = cat(1,varnames,included_gases{ig});
        aperrs = cat(1,aperrs,gascol_aperr(ig));
        gasnvar(ig) = 1;
        nv = nv + 1;
    end
    
end

% change the order of loops from Xiong's original code to handle different
% nalb for different windows
alb_idx_1 = -1;
alb_idx_2 = -1;
if inc_alb
    alb_idx_1 = nv;
    for iwin = 1:length(outp_d)
        for ialb = 1:outp_d(iwin).nalb
            varnames = cat(1,varnames,['a',num2str(ialb),'w',num2str(iwin)]);
            aperrs = cat(1,aperrs,albs_aperr(ialb));
            nv = nv+1;
        end
    end
    alb_idx_2 = nv-1;
end
% Add other terms
tidx = -1 ; assaidx = -1 ;
codidx = -1 ; cssaidx = -1 ; cfracidx = -1 ; sfcprsidx = -1;
airglowidx = -1;
if inc_t 
    varnames = cat(1,varnames,'T');
    aperrs = cat(1,aperrs,t_aperr);
    tidx = nv;
    nv = nv + 1;   
end

aodidxs = zeros(naerosol,1);
if inc_aod
    for ia = 1:naerosol
        aodidxs(ia) = nv;
        varnames = cat(1,varnames,[aerosols{ia},'-aod']);
        aperrs = cat(1,aperrs,aod_aperr(ia));
        nv = nv + 1;
    end
end

aodpkhidxs = zeros(naerosol,1);
if inc_aod_pkh
    for ia = 1:naerosol
        aodpkhidxs(ia) = nv;
        varnames = cat(1,varnames,[aerosols{ia},'-aod-pkh']);
        aperrs = cat(1,aperrs,aod_pkh_aperr(ia));
        nv = nv + 1;
    end
end

aodhfwidxs = zeros(naerosol,1);
if inc_aod_hfw
    for ia = 1:naerosol
        aodhfwidxs(ia) = nv;
        varnames = cat(1,varnames,[aerosols{ia},'-aod-hfw']);
        aperrs = cat(1,aperrs,aod_hfw_aperr(ia));
        nv = nv + 1;
    end
end

if inc_assa 
    varnames = cat(1,varnames,'assa');
    aperrs = cat(1,aperrs,assa_aperr);
    assaidx = nv;
    nv = nv + 1;   
end

if inc_cod 
    varnames = cat(1,varnames, 'cod');
    aperrs = cat(1,aperrs,cod_aperr);
    codidx = nv; 
    nv = nv + 1;   
end

if inc_cssa 
    varnames = cat(1,varnames,'cssa');
    aperrs = cat(1,aperrs,cssa_aperr);
    cssaidx = nv; 
    nv = nv + 1;   
end

if inc_cfrac
    varnames = cat(1,varnames,'cfrac');
    aperrs = cat(1,aperrs,cfrac_aperr);
    cfracidx = nv; 
    nv = nv + 1;   
end

if inc_sfcprs 
    varnames = cat(1,varnames,'sfcprs');
    aperrs = cat(1,aperrs,sfcprs_aperr);
    sfcprsidx = nv; 
    nv = nv + 1;   
end

if inc_airglow
    varnames = cat(1,varnames,'airglow');
    aperrs = cat(1,aperrs,airglow_aperr);
    airglowidx = nv; 
    nv = nv + 1;   
end

nv = nv-1;
% Derive a priori error covariance matrix 
% Assume correlation length of 6 km for profile retrieval
corrlen = 6.0;
sa = diag(aperrs.^2);
zmid = outp_d(1).zs;
for ig = 1: ngas
   if inc_prof(ig) 
      fidx = gasfidxs(ig);
      tempcovar = sa(fidx:fidx+nz-1, fidx:fidx+nz-1);
      for i = 1: nz
         for j = 1: i
            tempcovar(i, j) = sqrt(tempcovar(i, i) * tempcovar(j, j)) ...
                              * exp( -abs((zmid(i)-zmid(j))/corrlen));
            tempcovar(j, i) = tempcovar(i, j);
         end
      end
      sa(fidx:fidx+nz-1, fidx:fidx+nz-1) = tempcovar;
   end
end
% ==============================================
% set measurement vector and covariance matrix
% ==============================================
ny = length(rad);
% ymeas = log(rad);
ysig  = wsnr;
syn1 = diag(ysig.^2);

% ==============================================
% set measurement weighting functions and
%     a priori covariance matrix
% ==============================================
ywf = nan(ny,nv);

for ig = 1:ngas
    fidx = gasfidxs(ig);
    if inc_prof(ig)
        ywf(:,fidx:fidx+nz-1) = gasjac(:,:,ig);
    else
        ywf(:,fidx) = gascoljac(:,ig);
    end
end

% albedo terms
if inc_alb
    ywf(:,alb_idx_1:alb_idx_2) = albjac;
end

if tidx > 0
    ywf(:,tidx) = tjac;
end

if inc_aod
    for ia = 1:naerosol
        fidx = aodidxs(ia);
        ywf(:,fidx) = aodjac(:,ia);
    end
end

if inc_aod_pkh
    for ia = 1:naerosol
        fidx = aodpkhidxs(ia);
        ywf(:,fidx) = aodpkhjac(:,ia);
    end
end

if inc_aod_hfw
    for ia = 1:naerosol
        fidx = aodhfwidxs(ia);
        ywf(:,fidx) = aodhfwjac(:,ia);
    end
end

if assaidx > 0
    ywf(:,assaidx) = assajac;
end

if codidx > 0
    ywf(:,codidx) = codjac;
end

if cssaidx > 0
    ywf(:,cssaidx) = cssajac;
end

if cfracidx > 0
    ywf(:,cfracidx) = cfracjac;
end

if sfcprsidx > 0
    ywf(:,sfcprsidx) = sfcprsjac;
end

if airglowidx > 0
    ywf(:,airglowidx) = airglowjac;
end
% Apply OE 
ywf = double(ywf);
ywft = ywf';
sa = double(sa);
syn1 = double(syn1);

san1 = inv(sa);
temp = ywft * syn1 * ywf;
se = inv(temp + san1);
contri = se * ywft * syn1;
ak = contri * ywf;
sn  = (contri/(syn1)) * (contri)';
ss  = (ak - eye(nv)) * sa * (ak - eye(nv))';

h = outp_d(1).aircol/sum(outp_d(1).aircol);
h = h(:);
k = zeros(size(ak,1),1);
k(1:length(h)) = h;
outp.h = h;
outp.k = k;

outp.xch4e_a = sqrt(k'*sa*k);
xch4e_m = sqrt(k'*sn*k);
xch4e_s = sqrt(k'*ss*k);
xch4e_r = sqrt(k'*se*k);

outp.xch4e_m = xch4e_m;
outp.xch4e_s = xch4e_s;
outp.xch4e_r = xch4e_r;
if inc_airglow
    xch4e_f_airglow = 0;
    A_ue = ak(gasfidxs(1):(gasfidxs(2)-1),airglowidx);
    s_i_airglow = A_ue * airglow_aperr^2 * A_ue';
    xch4e_i_airglow = sqrt(h'*s_i_airglow*h);
else
    s_f_airglow = contri * double(airglowjac) * airglow_aperr^2 * double(airglowjac') * contri';
    xch4e_f_airglow = sqrt(k'*s_f_airglow*k);
    xch4e_i_airglow = 0;
end
outp.xch4e_f_airglow = xch4e_f_airglow;
outp.xch4e_i_airglow = xch4e_i_airglow;
if inc_t
    xch4e_f_t = 0;
    A_ue = ak(gasfidxs(1):(gasfidxs(2)-1),tidx);
    s_i_t = A_ue * t_aperr^2 * A_ue';
    xch4e_i_t = sqrt(h'*s_i_t*h);
else
    s_f_t = contri * double(tjac) * t_aperr^2 * double(tjac') * contri';
    xch4e_f_t = sqrt(k'*s_f_t*k);
    xch4e_i_t = 0;
end

outp.xch4e_f_t = xch4e_f_t;
outp.xch4e_i_t = xch4e_i_t;

if inc_sfcprs
    xch4e_f_sfcprs = 0;
    A_ue = ak(gasfidxs(1):(gasfidxs(2)-1),sfcprsidx);
    s_i_sfcprs = A_ue * sfcprs_aperr^2 * A_ue';
    xch4e_i_sfcprs = sqrt(h'*s_i_sfcprs*h);
else
    s_f_sfcprs = contri * double(sfcprsjac) * sfcprs_aperr^2 * double(sfcprsjac') * contri';
    xch4e_f_sfcprs = sqrt(k'*s_f_sfcprs*k);
    xch4e_i_sfcprs = 0;
end

outp.xch4e_f_sfcprs = xch4e_f_sfcprs;
outp.xch4e_i_sfcprs = xch4e_i_sfcprs;

if inc_aod
    xch4e_f_aod = 0;
    A_ue = ak(gasfidxs(1):(gasfidxs(2)-1),aodidxs);
    s_i_aod = A_ue * diag(aod_aperr(1:naerosol).^2) * A_ue';
    xch4e_i_aod = sqrt(h'*s_i_aod*h);
else
    s_f_aod = contri * double(aodjac) * diag(aod_aperr(1:naerosol).^2) * double(aodjac') * contri';
    xch4e_f_aod = sqrt(k'*s_f_aod*k);
    xch4e_i_aod = 0;
end

outp.xch4e_f_aod = xch4e_f_aod;
outp.xch4e_i_aod = xch4e_i_aod;

if inc_aod_pkh
    xch4e_f_aod_pkh = 0;
    A_ue = double(ak(gasfidxs(1):(gasfidxs(2)-1),aodpkhidxs));
    s_i_aod_pkh = A_ue * diag(aod_pkh_aperr(1:naerosol).^2) * A_ue';
    xch4e_i_aod_pkh = sqrt(h'*s_i_aod_pkh*h);
else
    s_f_aod_pkh = contri * double(aodpkhjac) * diag(aod_pkh_aperr(1:naerosol).^2) * double(aodpkhjac') * contri';
    xch4e_f_aod_pkh = sqrt(k'*s_f_aod_pkh*k);
    xch4e_i_aod_pkh = 0;
end

outp.xch4e_f_aod_pkh = xch4e_f_aod_pkh;
outp.xch4e_i_aod_pkh = xch4e_i_aod_pkh;

if inc_aod_hfw
    xch4e_f_aod_hfw = 0;
    A_ue = double(ak(gasfidxs(1):(gasfidxs(2)-1),aodhfwidxs));
    s_i_aod_hfw = A_ue * diag(aod_hfw_aperr(1:naerosol).^2) * A_ue';
    xch4e_i_aod_hfw = sqrt(h'*s_i_aod_hfw*h);
else
    s_f_aod_hfw = contri * double(aodhfwjac) * diag(aod_hfw_aperr(1:naerosol).^2) * double(aodhfwjac') * contri';
    xch4e_f_aod_hfw = sqrt(k'*s_f_aod_hfw*k);
    xch4e_i_aod_hfw = 0;
end

outp.xch4e_f_aod_hfw = xch4e_f_aod_hfw;
outp.xch4e_i_aod_hfw = xch4e_i_aod_hfw;
% sqrt(sum(sum(se(gasfidxs(1):gasfidxs(2)-1,gasfidxs(1):gasfidxs(2)-1))))/gastcol(1)

outp.ny = ny; outp.nv = nv; outp.varnames = varnames;
outp.ak = ak; outp.se = se; outp.sn = sn;
outp.ss = ss; outp.contri = contri;
outp.sa = sa; outp.aperrs = aperrs;
outp.ywf = ywf;

outp.gasfidxs = gasfidxs; 
outp.alb_idx_1 = alb_idx_1; outp.alb_idx_2 = alb_idx_2;
outp.tidx = tidx;
outp.aodidxs = aodidxs;
outp.assaidx = assaidx;
outp.codidx = codidx;
outp.cssaidx = cssaidx;
outp.cfracidx = cfracidx;
outp.sfcprsidx = sfcprsidx;

outp.gascoljac = gascoljac;
outp.inc_prof = inc_prof;
outp.included_gases = included_gases;

outp.nz = outp_d(1).nz;
outp.aircol = outp_d(1).aircol;
outp.ps = outp_d(1).ps;
outp.ts = outp_d(1).ts;
outp.zs = outp_d(1).zs;
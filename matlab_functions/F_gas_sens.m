function outp = F_gas_sens(inp,outp_d)
% Working horse. Matlab function to perform sensitivty study
% Modified from Xiong Liu's IDL subroutine gas_sens
% Takes the output from F_degrade_gc_output.m, outp_d, and its own input, inp

% Wrtitten by Kang Sun on 2017/09/05

% Rewritten by Kang Sun on 2017/09/06. The input now contains outp_d, the
% merged, degraded gc tool outputs, to make it work for multiple windows

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

gasnorm = ones(length(included_gases),1);

for i = 1:length(included_gases)
    new = '    ';
    new(1:length(included_gases{i})) = included_gases{i};
    if ~ismember(new,all_gases)
        error(['The molecule ',included_gases{i},' you asked for was not simulated by GC tool!!!'])
    end
end

if isfield(inp,'inc_prof')
    inc_prof = inp.inc_prof;
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
ngas = length(included_gases);
gasjac = []; 
gascoljac = [];
albjac = []; 
tjac = []; 
aodjac = []; 
assajac = []; 
codjac = []; 
cssajac = []; 
cfracjac = [];
sfcprsjac = [];
wsnr = [];
rad = [];

nz = outp_d(1).nz;
gascol = zeros(nz,ngas);
gastcol = zeros(ngas,1);
% number of albedo terms for all windows
nalb_all = 0;
for iwin = 1:length(outp_d);
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
                gasnorm(ig) = outp_d(iwin).gasnorm(igc);
                gascol(:,ig) = outp_d(iwin).gascol(:,igc);
                gastcol(ig) = outp_d(iwin).gastcol(igc);
                tmp_col_jac(:,ig) = outp_d(iwin).gascol_jac(:,igc);
                if inc_prof(ig)
                    tmp_prof_jac(:,:,ig) = outp_d(iwin).gas_jac(:,:,igc);
                end
            end
        end
    end
    gascoljac = cat(1,gascoljac,tmp_col_jac);
    gasjac = cat(1,gasjac,tmp_prof_jac);
    wsnr = cat(1,wsnr,outp_d(iwin).wsnr);
    rad = cat(1,rad,outp_d(iwin).rad);
    
    tmp_albjec = zeros(outp_d(iwin).nw,nalb_all);
    tmp_albjec(:,alb_count:alb_count+outp_d(iwin).nalb-1) = outp_d(iwin).surfalb_jac;
    alb_count = alb_count+outp_d(iwin).nalb;
    if inc_alb
        albjac = cat(1,albjac,tmp_albjec);
    end
    
    if inc_t
        tjac = cat(1,tjac,outp_d(iwin).t_jac);
    end
    if inc_sfcprs
        sfcprsjac = cat(1,sfcprsjac,outp_d(iwin).sfcprs_jac);
    end
    if inc_aod
        aodjac = cat(1,aodjac,outp_d(iwin).aod_jac);
    end
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
outp.rad = rad;
outp.wsnr = wsnr;
outp.gasnorm = gasnorm;
outp.gascol = gascol;
outp.gastcol = gastcol;

% set up state vectors, and a priori error
nv = 1;% note the difference indexing between matlab and idl
aperrs = [];   %no need remove first value later
varnames = cell(0); %no need remove first value later

gasfidxs = zeros(ngas,1);
gasnvar  = zeros(ngas,1);

% construct gases a priori
gascol_aperr = inp.gascol_aperr_scale*gastcol; 

gasprof_aperr_scale = zeros(outp_d(1).nz,1); 
gasprof_aperr_scale(outp_d(1).zmid <= 2) = inp.gasprof_aperr_scale_LT;
gasprof_aperr_scale(outp_d(1).zmid > 2 & outp_d(1).zmid <= 17) = inp.gasprof_aperr_scale_UT;
gasprof_aperr_scale(outp_d(1).zmid > 17) = inp.gasprof_aperr_scale_ST;
gasprof_aperr = gascol.*repmat(gasprof_aperr_scale,[1,ngas]);
outp.gasprof_aperr = gasprof_aperr;

albs_aperr = inp.albs_aperr; 
t_aperr = inp.t_aperr; 
aod_aperr = inp.aod_aperr;
assa_aperr = inp.assa_aperr;
cod_aperr = inp.cod_aperr;
cssa_aperr = inp.cssa_aperr;
cfrac_aperr = inp.cfrac_aperr;
sfcprs_aperr = inp.sfcprs_aperr;

for ig = 1: ngas
    gasfidxs(ig) = nv;
    
    if inc_prof(ig)
        tempcell = cell(nz,1);
        for icell = 1:nz
            tempcell{icell} = [included_gases{ig},'_',num2str(icell)];
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
tidx = -1 ; aodidx = -1 ; assaidx = -1 ;
codidx = -1 ; cssaidx = -1 ; cfracidx = -1 ; sfcprsidx = -1;
if inc_t 
    varnames = cat(1,varnames,'T');
    aperrs = cat(1,aperrs,t_aperr);
    tidx = nv;
    nv = nv + 1;   
end

if inc_aod
    varnames = cat(1,varnames,'aod');
    aperrs = cat(1,aperrs,aod_aperr);
    aodidx = nv;
    nv = nv + 1;  
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
nv = nv-1;
% Derive a priori error covariance matrix 
% Assume correlation length of 6 km for profile retrieval
corrlen = 6.0;
sa = diag(aperrs.^2);
zmid = outp_d(1).zmid;
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

if aodidx > 0
    ywf(:,aodidx) = aodjac;
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

outp.ny = ny; outp.nv = nv; outp.varnames = varnames;
outp.ak = ak; outp.se = se; outp.sn = sn;
outp.ss = ss; outp.contri = contri;
outp.sa = sa; outp.aperrs = aperrs;
outp.ywf = ywf;

outp.gasfidxs = gasfidxs; 
outp.alb_idx_1 = alb_idx_1; outp.alb_idx_2 = alb_idx_2;
outp.tidx = tidx;
outp.aodidx = aodidx;
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
outp.zmid = outp_d(1).zmid;
outp.tmid = outp_d(1).tmid;

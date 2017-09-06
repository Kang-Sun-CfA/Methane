function outp = F_gas_sens(inp,outp_gc)
% Working horse. Matlab function to perform sensitivty study
% Modified from Xiong Liu's IDL subroutine gas_sens
% Takes the output from F_read_gc_output.m, outp_gc, and its own input, inp

% Wrtitten by Kang Sun on 2017/09/05

nwin = inp.nwin; wmins = inp.wmins; wmaxs = inp.wmaxs;
fwhm = inp.fwhm; nsamp = inp.nsamp;
snr = inp.snr; snrdefine_rad = inp.snrdefine_rad;

nz = outp_gc.nz; zmid = outp_gc.zmid;
ngas = outp_gc.ngas; gases = outp_gc.gases;
nw0 = outp_gc.nw; wave0 = outp_gc.wave;
rad0 = outp_gc.rad; irrad0 = outp_gc.irrad;

if isfield(inp,'inc_gas')
    inc_gas = inp.inc_gas;
else
    inc_gas = zeros(1,ngas);
end
if isfield(inp,'inc_prof')
    inc_prof = inp.inc_prof;
else
    inc_prof = zeros(1,ngas);
end
if isfield(inp,'inc_alb')
    inc_alb = inp.inc_alb;
    nalb = inp.nalb;
else
    inc_alb = 0;
    nalb = 1;
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

gasjac0 = outp_gc.gas_jac; gascoljac0 = outp_gc.gascol_jac;
albjac0 = outp_gc.surfalb_jac; tjac0 = outp_gc.t_jac; 
aodjac0 = outp_gc.aod_jac; assajac0 = outp_gc.assa_jac; 
codjac0 = outp_gc.cod_jac; cssajac0 = outp_gc.cssa_jac;
cfracjac0 = outp_gc.cfrac_jac;
sfcprsjac0 = outp_gc.sfcprs_jac;

% ==================================================
% Convole with slit function and/or sample spectra
% ==================================================
[nw, wave, rad, irrad, gasjac, gascoljac,  ...
    albjac, tjac, aodjac, assajac, codjac, cssajac, cfracjac, sfcprsjac, albjacs, ...
    wsnr, fidxs, lidxs, mnrads]...
    = multiwin_convol_samp_snr(snrdefine_rad, inc_t, inc_aod, inc_assa, inc_cod, inc_cssa, ...
    inc_cfrac, inc_sfcprs, nalb, nz, ngas, nw0, wave0, rad0, irrad0, albjac0, gasjac0,...
    gascoljac0, tjac0, aodjac0, assajac0, codjac0, cssajac0, cfracjac0,...
    sfcprsjac0, nwin, wmins, wmaxs, fwhm, snr, nsamp);

gasprof_aperr = inp.gasprof_aperr; 
gascol_aperr = inp.gascol_aperr;
albs_aperr = inp.albs_aperr; 
t_aperr = inp.t_aperr; 
aod_aperr = inp.aod_aperr;
assa_aperr = inp.assa_aperr;
cod_aperr = inp.cod_aperr;
cssa_aperr = inp.cssa_aperr;
cfrac_aperr = inp.cfrac_aperr;
sfcprs_aperr = inp.sfcprs_aperr;

% set up state vectors, and a priori error
nv = 1;% note the difference indexing between matlab and idl
aperrs = [];   %no need remove first value later
varnames = cell(0); %no need remove first value later

gasfidxs = zeros(ngas,1);
gasnvar  = zeros(ngas,1);

for ig = 1: ngas 
   gasfidxs(ig) = nv;
   
   if inc_gas(ig) == 1 
      if inc_prof(ig) == 1 
          tempcell = cell(nz,1);
          for icell = 1:nz
              tempcell{icell} = [gases{ig},num2str(icell)];
          end
         varnames = [varnames; tempcell];
         aperrs = [aperrs; gasprof_aperr(:, ig)];
         gasnvar(ig) = nz;
         nv = nv + nz;
      else
         varnames = [varnames; gases{ig}];
         aperrs = [aperrs; gascol_aperr(ig)];
         gasnvar(ig) = 1;
         nv = nv + 1;
      end
   end
end

albidxs = -ones(nalb, nwin);
if inc_alb == 1    
   for i = 1: nalb 
      for iwin = 1: nwin 
         albidxs(i, iwin) = nv;
         varnames = [varnames; ['a', num2str(i),'w', num2str(iwin)]];
         aperrs = [aperrs; albs_aperr(i)];
         nv = nv + 1;
      end
   end
end

% Add other terms
tidx=-1 ; aodidx = -1 ; assaidx=-1 ;
codidx = -1 ; cssaidx = -1 ; cfracidx = -1 ; sfcprsidx = -1;
if inc_t == 1 
    varnames = [varnames; 'T'];
    aperrs = [aperrs; t_aperr];
    tidx = nv;
    nv = nv + 1;   
end

if inc_aod == 1 
    varnames = [varnames; 'aod'];
    aperrs = [aperrs; aod_aperr];
    aodidx = nv;
    nv = nv + 1;  
end

if inc_assa == 1 
    varnames = [varnames; 'assa'];
    aperrs = [aperrs; assa_aperr];
    assaidx = nv;
    nv = nv + 1;   
end

if inc_cod == 1 
    varnames = [varnames; 'cod'];
    aperrs = [aperrs; cod_aperr];
    codidx = nv; 
    nv = nv + 1;   
end

if inc_cssa == 1 
    varnames = [varnames; 'cssa'];
    aperrs = [aperrs; cssa_aperr];
    cssaidx = nv; 
    nv = nv + 1;   
end

if inc_cfrac == 1 
    varnames = [varnames; 'cfrac'];
    aperrs = [aperrs; cfrac_aperr];
    cfracidx = nv; 
    nv = nv + 1;   
end

if inc_sfcprs == 1 
    varnames = [varnames; 'sfcprs'];
    aperrs = [aperrs; sfcprs_aperr];
    sfcprsidx = nv; 
    nv = nv + 1;   
end
nv = nv-1;
% varnames = strcompress(varnames(1:nv), /remove_all) % no idea what's this
% aperrs   = aperrs(1:nv)

% Derive a priori error covariance matrix 
% Assume correlation length of 6 km for profile retrieval
corrlen = 6.0;
sa = diag(aperrs.^2);

for ig = 1: ngas
   if inc_gas(ig) == 1 && inc_prof(ig) == 1 
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
ny = nw;
ymeas = [log(rad)];
ysig  = [wsnr];
syn1 = diag(ysig.^2);

% ==============================================
% set measurement weighting functions and
%     a priori covariance matrix
% ==============================================
ywf = nan(nw, nv);

for ig = 1:ngas
    if inc_gas(ig) == 1
        fidx = gasfidxs(ig);
        if inc_prof(ig) == 1
            ywf(:,fidx:fidx+nz-1) = gasjac(:,:,ig);
        else
            ywf(:,fidx) = gascoljac(:,ig);
        end
    end
end

% albedo terms
fidxs = fidxs+1; % note the difference indexing between matlab and idl
lidxs = lidxs+1;
if inc_alb == 1
    for i = 1:nalb
        for iwin = 1:nwin
            if albidxs(i,iwin) > 0
                fidx = fidxs(iwin); lidx = lidxs(iwin);
                ywf(fidx:lidx,albidxs(i,iwin)) = albjacs(i,fidx:lidx);
            end
        end
    end
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
% note the column/row difference between matlab and idl
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
outp = [];
outp.ny = ny;
outp.nv = nv;
outp.varnames = varnames;
outp.gasfidxs = gasfidxs;
outp.gasnvar = gasnvar;
outp.ak = ak;
outp.se = se;
outp.sn = sn;
outp.ss = ss;
outp.contri = contri;
outp.albidxs = albidxs;
outp.tidx = tidx;
outp.aodidx = aodidx;
outp.assaidx = assaidx;
outp.codidx = codidx;
outp.cssaidx = cssaidx;
outp.cfracidx = cfracidx;
outp.sfcprsidx = sfcprsidx;
outp.wave = wave;
outp.rad = rad;
outp.irrad = irrad;
outp.wsnr = wsnr;
outp.ywf = ywf;
outp.sa = sa;
outp.aperrs = aperrs;

function  [ nw, wave, rad, irrad, gasjac, gascoljac,  ...
    albjac, tjac, aodjac, assajac, codjac, cssajac, cfracjac, sfcprsjac, albjacs, ...
    wsnr, fidxs, lidxs, mnrads]...
    = multiwin_convol_samp_snr(snrdefine_rad, inc_t, inc_aod, inc_assa, inc_cod, inc_cssa, ...
    inc_cfrac, inc_sfcprs, nalb, nz, ngas, nw0, wave0, rad0, irrad0, albjac0, gasjac0,...
    gascoljac0, tjac0, aodjac0, assajac0, codjac0, cssajac0, cfracjac0,...
    sfcprsjac0, nwin, wmins, wmaxs, fwhm, snr, nsamp)

if fwhm > 0
    
    slit = fwhm/1.66511;% half width at 1e
    dw1 = fwhm/nsamp;
    
    dw0 = wave0(2)-wave0(1);
    ndx = ceil(slit*2.7/dw0);
    xx = (0:ndx*2)*dw0-ndx*dw0;
    kernel = exp(-(xx/slit).^2);
    kernel = kernel/sum(kernel);
    
    wavemin = wmins(1); wavemax = wmaxs(nwin);
    nw = round((wavemax-wavemin)/dw1+1);
    wave = wavemin+(0:nw-1)*dw1;
    % convolve and sample
    rad = dgrad_sample_spec(wave0, rad0, kernel, wave);
    irrad = dgrad_sample_spec(wave0, irrad0, kernel, wave);
    albjac = dgrad_sample_spec(wave0, albjac0, kernel, wave);
    
    gasjac = zeros(nw, nz, ngas);
    for iz = 1:nz
        for ig = 1:ngas
            tempjac = squeeze(gasjac0(:,iz,ig));
            gasjac(:,iz,ig) = dgrad_sample_spec(wave0, tempjac, kernel, wave);
        end
    end
    gascoljac = zeros(nw,ngas);
    for ig = 1:ngas
        tempjac = gascoljac0(:,ig);
        gascoljac(:,ig) = dgrad_sample_spec(wave0, tempjac, kernel, wave);
    end
    
    if inc_t == 1
        tjac = dgrad_sample_spec(wave0, tjac0, kernel, wave);
    else
        tjac = 0*wave;
    end
    
    if inc_aod == 1
        aodjac = dgrad_sample_spec(wave0, aodjac0, kernel, wave);
    else
        aodjac = 0*wave;
    end
    if inc_assa == 1
        assajac = dgrad_sample_spec(wave0, assajac0, kernel, wave);
    else
        assajac = 0*wave;
    end
    if inc_cod == 1
        codjac = dgrad_sample_spec(wave0, codjac0, kernel, wave);
    else
        codjac = 0*wave;
    end
    if inc_cssa == 1
        cssajac = dgrad_sample_spec(wave0, cssajac0, kernel, wave);
    else
        cssajac = 0*wave;
    end
    
    if inc_cfrac == 1
        cfracjac = dgrad_sample_spec(wave0, cfracjac0, kernel, wave);
    else
        cfracjac = 0*wave;
    end
    if inc_sfcprs == 1
        sfcprsjac = dgrad_sample_spec(wave0, sfcprsjac0, kernel, wave);
    else
        sfcprsjac = 0*wave;
    end
    
    % subset the data
    [nw, wave, rad, irrad, gasjac, gascoljac, tjac, aodjac, ...
        assajac, codjac, cssajac, cfracjac, albjac, sfcprsjac] = ...
        subset_waverange_ncpy(nwin, wmins, wmaxs, wave, rad, irrad, ...
        gasjac, gascoljac, tjac, aodjac, assajac, codjac, cssajac, ...
        cfracjac, albjac, sfcprsjac);
else
    nw = nw0;
    wave = wave0;
    rad = rad0;
    irrad = irrad0;
    gasjac = gasjac0;
    gascoljac = gascoljac0;
    albjac = albjac0;
    tjac = tjac0;
    aodjac  = aodjac0;
    assajac = assajac0;
    codjac = codjac0;
    cssajac=  cssajac0;
    cfracjac = cfracjac0;
    sfcprsjac = sfcprsjac0;
    % subset the data
    [nw, wave, rad, irrad, gasjac, gascoljac, tjac, aodjac, ...
        assajac, codjac, cssajac, cfracjac, albjac, sfcprsjac] = ...
        subset_waverange_ncpy(nwin, wmins, wmaxs, wave, rad, irrad, ...
        gasjac, gascoljac, tjac, aodjac, assajac, codjac, cssajac, ...
        cfracjac, albjac, sfcprsjac);
end
% derive albedo polynomial jacs
fidxs = zeros(nwin,1); lidxs = fidxs;
mnrads = fidxs;
albjacs = zeros(nalb, nw);
albjacs(1,:) = albjac;
fidx = 0;
for iwin = 1:nwin
    da = wave > wmins(iwin) & wave < wmaxs(iwin);
    ntemp = sum(da);
    lidx = fidx+ntemp-1;
    fidxs(iwin) = fidx; lidxs(iwin) = lidx;
    
    wavg = mean(wave(da));
    wavedf = wave(da)-wavg;
    
    for i = 1:nalb-1
        albjacs(i+1,da) = albjacs(i,da).*wavedf(:)';
    end
    mnrads(iwin) = mean(rad(da));
    fidx = lidx+1;
end
% Assign SNR to measured radiance

wsnr = snr * sqrt(rad / snrdefine_rad);
return

function rad1 = dgrad_sample_spec(wave, rad, kernel, wave1)

rad1 = conv(rad, kernel, 'same');
rad1 = spline(wave, rad1, wave1);
rad1 = rad1(:);
return

function [nw, wave1, rad1, irrad1, gasjac1, gascoljac1, tjac1, aodjac1, ...
    assajac1, codjac1, cssajac1, cfracjac1, albjac1, sfcprsjac1] = ...
    subset_waverange_ncpy(nwin, wmins, wmaxs, wave, rad, irrad, ...
      gasjac, gascoljac, tjac, aodjac, assajac, codjac, cssajac, ...
      cfracjac, albjac, sfcprsjac)

selw = zeros(size(wave));
for iwin = 1: nwin
    wmin = wmins(iwin);
    wmax = wmaxs(iwin);
    
    temp = (wave > wmin & wave < wmax);
    selw = selw | temp;
end
nw = sum(selw);

if nw > 0
    wave1        = wave(selw);
    rad1         = rad(selw);
    irrad1       = irrad(selw);
    gasjac1      = gasjac(selw, :, :);
    gascoljac1  = gascoljac(selw,:);
    tjac1       = tjac(selw);
    aodjac1     = aodjac(selw);
    assajac1    = assajac(selw);
    codjac1     = codjac(selw);
    cssajac1    = cssajac(selw);
    cfracjac1   = cfracjac(selw);
    albjac1     = albjac(selw);
    sfcprsjac1  = sfcprsjac(selw);
else
    disp('No wavelength subsetted!!!')    
end
return
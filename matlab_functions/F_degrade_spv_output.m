function outp = F_degrade_spv_output(inp)
% Convolve the output from spv with a gaussian FWHM. The output is a structure
% array with each element corresponds to a window. Modified from
% F_degrade_gc_output.m by Kang Sun on 2018/08/20
% updated the instrument model on 2018/11/08 and add ILS jacobians

outp = struct;
nwin = inp.nwin; wmins = inp.wmins; wmaxs = inp.wmaxs;
if ~isfield(inp,'fwhm')
    inp.fwhm  = nan(nwin,1);
    for iwin = 1:nwin
        inp.fwhm(iwin) = (inp.wmaxs(iwin)-inp.wmins(iwin))/inp.npixel(iwin)*inp.nsamp(iwin);
    end
end
fwhm = inp.fwhm; nsamp = inp.nsamp;
nalb = inp.nalb;
% snre = inp.snre;
% snrdefine_rad = inp.snrdefine_rad;
% snrdefine_dlambda = inp.snrdefine_dlambda;

if ~isfield(inp,'inpn')
    error('You need to specify inputs for the noise model!')
end
inpn = inp.inpn;

if ~isfield(inp,'inp_ils')
    warning('No ILS is specified. Use Gaussian.')
    inp_ils = [];
else
    inp_ils = inp.inp_ils;
end


spv_fwhm = zeros(nwin,1);
for iwin = 1:nwin
    clear spv_output
    for igc = 1:length(inp.spv_output)
        if wmins(iwin) > inp.spv_output{igc}.wave(1) && wmaxs(iwin) < inp.spv_output{igc}.wave(end)
            spv_output = inp.spv_output{igc};
        end
    end
    if ~exist('spv_output','var');error('Your window is out of range!!!');end
    if isfield(inp,'spv_fwhm')
        spv_fwhm(iwin) = inp.spv_fwhm(iwin);
    else
        spv_fwhm(iwin) = median(diff(spv_output.wave))*2;
    end
    % just in case the gc fwhm is not negligible
    fwhm_true = sqrt(fwhm(iwin)^2-spv_fwhm(iwin)^2);
    % w2 is the low resolution wavelength grid
    w2 = wmins(iwin):fwhm(iwin)/nsamp(iwin):wmaxs(iwin);
    % w1 is the high resolution wavelength grid
    w1 = spv_output.wave;
    fieldn = fieldnames(spv_output);
    % convolve all fields with length equal to nw to the specified fwhm
    for ifield = 1:length(fieldn)
        if ~iscell(spv_output.(fieldn{ifield}))
            if size(spv_output.(fieldn{ifield}),1) == spv_output.nw
                s1 = spv_output.(fieldn{ifield});
                if ~isempty(inp_ils)
                    outp(iwin).(fieldn{ifield}) = F_instrument_model(w1,s1,fwhm_true,w2,inp_ils{iwin});
                else
                    outp(iwin).(fieldn{ifield}) = F_instrument_model(w1,s1,fwhm_true,w2,inp_ils);
                end
            else
                outp(iwin).(fieldn{ifield}) = spv_output.(fieldn{ifield});
            end
        else
            outp(iwin).(fieldn{ifield}) = spv_output.(fieldn{ifield});
        end
    end
    
    % calculate wavelength shift jacobians
    s1 = spv_output.rad;
    dstep = 1e-5;
    if ~isempty(inp_ils)
        outp(iwin).shift_jac = (F_instrument_model(w1,s1,fwhm_true,w2+dstep,inp_ils{iwin})-...
            F_instrument_model(w1,s1,fwhm_true,w2,inp_ils{iwin}))/dstep;
    else
        outp(iwin).shift_jac = (F_instrument_model(w1,s1,fwhm_true,w2+dstep,inp_ils)-...
            F_instrument_model(w1,s1,fwhm_true,w2,inp_ils))/dstep;
    end
    % zlo (in the unit of 2e13) jacobians
    outp(iwin).zlo_jac = ones(size(outp(iwin).rad))*2e13;
    
    outp(iwin).nw = length(outp(iwin).wave);
    if ~isempty(inp_ils)
        if ~isfield(inp_ils{iwin},'do_jac')
            do_jac = false;
        else
            do_jac = inp_ils{iwin}.do_jac;
        end
        if do_jac
            dstep = 5e-4;
            inp_ils_tmp = inp_ils{iwin};
            inp_ils_tmp.m = inp_ils{iwin}.m*(1-dstep);
            rad1 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            inp_ils_tmp.m = inp_ils{iwin}.m*(1+dstep);
            rad2 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            outp(iwin).ils_m_jac = (rad2-rad1)/(2*dstep);
            
            % jac for eta is not dy/dlnx, it is dy/dx
            inp_ils_tmp = inp_ils{iwin};
            inp_ils_tmp.eta = inp_ils{iwin}.eta-dstep;
            rad1 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            inp_ils_tmp.eta = inp_ils{iwin}.eta+dstep;
            rad2 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            outp(iwin).ils_eta_jac = (rad2-rad1)/(2*dstep);
            
            inp_ils_tmp = inp_ils{iwin};
            inp_ils_tmp.k = inp_ils{iwin}.k*(1-dstep);
            rad1 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            inp_ils_tmp.k = inp_ils{iwin}.k*(1+dstep);
            rad2 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            outp(iwin).ils_k_jac = (rad2-rad1)/(2*dstep);
            
            % jac for aw is not dy/dlnx, it is dy/dx
            inp_ils_tmp = inp_ils{iwin};
            inp_ils_tmp.aw = inp_ils{iwin}.aw-dstep;
            rad1 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            inp_ils_tmp.aw = inp_ils{iwin}.aw+dstep;
            rad2 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true,double(w2),inp_ils_tmp);
            outp(iwin).ils_aw_jac = (rad2-rad1)/(2*dstep);
            
            rad1 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true*(1-dstep),double(w2),inp_ils{iwin});
            rad2 = F_instrument_model(double(spv_output.wave),double(spv_output.rad),fwhm_true*(1+dstep),double(w2),inp_ils{iwin});
            outp(iwin).ils_fwhm_jac = (rad2-rad1)/(2*dstep);
            
        end
    end
    %     outp(iwin).gases = spv_output.gases;
    %     outp(iwin).gasnorm = spv_output.gasnorm;
end
% derive albedo polynomial jacs
fidxs = zeros(nwin,1); lidxs = fidxs;
% mnrads = fidxs;
fidx = 0;
for iwin = 1:nwin
    albjacs = zeros(length(outp(iwin).wave),nalb(iwin));
    albjacs(:,1) = outp(iwin).surfalb_jac;
    ntemp = length(outp(iwin).wave);
    lidx = fidx+ntemp-1;
    fidxs(iwin) = fidx; lidxs(iwin) = lidx;
    
    wavg = mean(outp(iwin).wave);
    wavedf = outp(iwin).wave-wavg;
    
    for i = 1:nalb(iwin)-1
        albjacs(:,i+1) = albjacs(:,i).*wavedf(:);
    end
    outp(iwin).surfalb_jac = albjacs;
    %     mnrads(iwin) = mean(rad);
    fidx = lidx+1;
    % Assign SNR to measured radiance
    
    inpn{iwin}.wave = outp(iwin).wave;
    inpn{iwin}.I = outp(iwin).rad;
    inpn{iwin}.dl = fwhm(iwin)/nsamp(iwin);
    outpn = F_noise_spv(inpn{iwin});
    outp(iwin).wsnr = outpn.wsnr;
    outp(iwin).wsnr_shot = outpn.wsnr_shot;
    outp(iwin).wsnr_single = outpn.wsnr_single;
    outp(iwin).S = outpn.S;
    outp(iwin).nalb = nalb(iwin);
end
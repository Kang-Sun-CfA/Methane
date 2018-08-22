function outp = F_degrade_spv_output(inp)
% Convolve the output from spv with a gaussian FWHM. The output is a structure
% array with each element corresponds to a window. Modified from
% F_degrade_gc_output.m by Kang Sun on 2018/08/20

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
                outp(iwin).(fieldn{ifield}) = F_conv_interp_n(w1,s1,fwhm_true,w2);
            else
                outp(iwin).(fieldn{ifield}) = spv_output.(fieldn{ifield});
            end
        else
            outp(iwin).(fieldn{ifield}) = spv_output.(fieldn{ifield});
        end
    end
    outp(iwin).nw = length(outp(iwin).wave);
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
    
    outp(iwin).nalb = nalb(iwin);
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
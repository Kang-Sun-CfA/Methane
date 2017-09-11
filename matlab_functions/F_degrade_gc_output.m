function outp = F_degrade_gc_output(inp)
% Convolve the output from GC tool with a gaussian FWHM. Major upgrade from
% Xiong's code to differentiate multiple windows. The output is a structure
% array with each element corresponds to a window.

% Written by Kang Sun on 2017/09/06

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
snre = inp.snre;
snrdefine_rad = inp.snrdefine_rad;
snrdefine_dlambda = inp.snrdefine_dlambda;
if isfield(inp,'gc_fwhm');
    gc_fwhm = inp.gc_fwhm;
else
    gc_fwhm = 0;
end

for iwin = 1:nwin
    clear gc_output
    for igc = 1:length(inp.gc_output)
        if wmins(iwin) > inp.gc_output{igc}.wave(1) && wmaxs(iwin) < inp.gc_output{igc}.wave(end)
            gc_output = inp.gc_output{igc};
        end
    end
    if ~exist('gc_output','var');error('Your window is out of range!!!');end
    % just in case the gc fwhm is not negligible
    fwhm_true = sqrt(fwhm(iwin)^2-gc_fwhm^2);
    % w2 is the low resolution wavelength grid
    w2 = wmins(iwin):fwhm(iwin)/nsamp(iwin):wmaxs(iwin);
    % w1 is the high resolution wavelength grid
    w1 = gc_output.wave;
    fieldn = fieldnames(gc_output);
    % convolve all fields with length equal to nw to the specified fwhm
    for ifield = 1:length(fieldn)
        if ~iscell(gc_output.(fieldn{ifield}))
            if size(gc_output.(fieldn{ifield}),1) == gc_output.nw
                s1 = gc_output.(fieldn{ifield});
                outp(iwin).(fieldn{ifield}) = F_conv_interp_n(w1,s1,fwhm_true,w2);
            else
                outp(iwin).(fieldn{ifield}) = gc_output.(fieldn{ifield});
            end
        else
            outp(iwin).(fieldn{ifield}) = gc_output.(fieldn{ifield});
        end
    end
    outp(iwin).nw = length(outp(iwin).wave);
    %     outp(iwin).gases = gc_output.gases;
    %     outp(iwin).gasnorm = gc_output.gasnorm;
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
    outp(iwin).wsnr = snre(iwin) * ...
        sqrt(outp(iwin).rad / snrdefine_rad(iwin) * ...
        fwhm(iwin)/nsamp(iwin) / snrdefine_dlambda(iwin));
    outp(iwin).nalb = nalb(iwin);
end
return

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
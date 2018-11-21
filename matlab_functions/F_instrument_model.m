function s1_low = F_instrument_model(w1,s1,fwhm,common_grid,inp_ils)
% This function convolves s1 with a specified fwhm, resample it to
% common_grid

% Rewritten by Kang Sun from on F_conv_interp_n.m on 2018/11/08 to define
% ILS shapes other than Gaussian

if isempty(inp_ils) 
if ~isfield(inp_ils,'ils_extent')
    ils_extent = 2.7;
else
    ils_extent = inp_ils.ils_extent;
end
else
    ils_extent = 2.7;
end
slit = fwhm/1.66511;% half width at 1e
dw0 = median(diff(w1));
ndx = ceil(slit*ils_extent/dw0);
xx = (0:ndx*2)*dw0-ndx*dw0;
if isempty(inp_ils) 
    kernel = exp(-(xx/slit).^2);
    kernel = kernel/sum(kernel);
else
    inp_ils.fwhm = fwhm;
    inp_ils.xx = xx;
    switch inp_ils.ils_shape
        case 'SG_P7'
            kernel = F_ils_SG_P7(inp_ils);
            kernel = kernel/sum(kernel);
        otherwise
            kernel = exp(-(xx/slit).^2);
            kernel = kernel/sum(kernel);
    end
end
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


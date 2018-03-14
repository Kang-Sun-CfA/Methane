function outp = F_merge_gc_output_multi(inp)
% Merge multiple outputs from GC tool into one. The gases have to be the
% same. The wavelength interval has to be the same, and in phase.

% modifed from F_merge_gc_output.m by Kang Sun on 2018/03/14 to enable more
% than 3 windows

nwin = length(inp.fn_cell);
outp_cell = cell(nwin,1);
inp0 = inp;
for iwin = 1:nwin
    inp0.fn = inp.fn_cell{iwin};
outp_cell{iwin} = F_read_gc_output(inp0);
if ~isequal(outp_cell{iwin}.gases,outp_cell{1}.gases)
    error('Gases have to be the SAME!!!')
end
end

outp = outp_cell{1};
for iwin = 1:nwin-1
int1 = outp.wave < mean([max(outp.wave),min(outp_cell{iwin+1}.wave)]);
int2 = outp_cell{iwin+1}.wave >= mean([max(outp.wave),min(outp_cell{iwin+1}.wave)]);

fieldn = fieldnames(outp);

for i = 1:length(fieldn)
    if ~iscell(outp.(fieldn{i}))
        if size(outp.(fieldn{i}),1) == outp.nw
            tmp1 = outp.(fieldn{i})(int1,:,:);
            tmp2 = outp_cell{iwin+1}.(fieldn{i})(int2,:,:);
            outp.(fieldn{i}) = cat(1,tmp1,tmp2);
        end
    end
end
outp.nw = length(outp.wave);
end


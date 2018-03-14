function outp = F_merge_gc_output(inp)
% Merge multiple outputs from GC tool into one. The gases have to be the
% same. The wavelength interval has to be the same, and in phase.

% Written by Kang Sun on 2017/09/06
% updated on 2018/03/14 to enable merging airglow windows

fn1 = inp.fn1;
fn2 = inp.fn2;
inp1.fn = fn1;
inp2.fn = fn2;
if isfield(inp,'airglowspec_path')
    inp1.airglowspec_path = inp.airglowspec_path;
    inp1.airglowspec_path = inp.airglowspec_path;
end
if isfield(inp,'VZA')
    inp1.VZA = inp.VZA;
    inp1.VZA = inp.VZA;
end
outp1 = F_read_gc_output(inp1);
outp2 = F_read_gc_output(inp2);

if ~isequal(outp1.gases,outp2.gases)
    error('Gases have to be the SAME!!!')
end

int1 = outp1.wave < mean([max(outp1.wave),min(outp2.wave)]);
int2 = outp2.wave >= mean([max(outp1.wave),min(outp2.wave)]);
outp = outp1;

fieldn = fieldnames(outp1);

for i = 1:length(fieldn)
    if ~iscell(outp1.(fieldn{i}))
        if size(outp1.(fieldn{i}),1) == outp1.nw
            tmp1 = outp1.(fieldn{i})(int1,:,:);
            tmp2 = outp2.(fieldn{i})(int2,:,:);
            outp.(fieldn{i}) = cat(1,tmp1,tmp2);
        end
    end
end
outp.nw = length(outp.wave);
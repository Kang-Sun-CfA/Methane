function outp = F_write_GCtool_input(inp)
% Sadly unix(command) does not work in matlab when aerosols are added

% Written by Kang Sun 2017/09/05
% add the option to change input profiles on 2017/10/11
sfs = filesep;

if isfield(inp,'SZA')
    SZA = inp.SZA;
else
    SZA = 70;
end

if isfield(inp,'VZA')
    VZA = inp.VZA;
else
    VZA = 45;
end

vStart = inp.vStart; vEnd = inp.vEnd;

gas_cell = inp.gas_cell;
ngas = length(gas_cell);
gas_str = strjoin(gas_cell);

dv_calc = inp.dv_calc;

if isfield(inp,'input_profile')
    input_profile = inp.input_profile;
else
    input_profile = '../new_input/input_clean.asc';
end

if isfield(inp,'dv_out')
    dv_out = inp.dv_out;
else
    dv_out = dv_calc;
end
if isfield(inp,'FWHM')
    FWHM = inp.FWHM;
else
    FWHM = 2*dv_calc;
end

if isfield(inp,'ReflSpectra')
    ReflSpectra = inp.ReflSpectra;
else
    ReflSpectra = 'grass_ASTER.dat';
end

if isfield(inp,'K1')
    K1 = inp.K1; K2 = inp.K2; K3 = inp.K3;
else
    K1 = 0.1; K2 = 0.1; K3 = 0.00001;
end

if isfield(inp,'fn_extra')
    fn_extra = inp.fn_extra;
else
    fn_extra = 'test';
end
fn_pre = ['.',sfs,'outp',sfs,gas_cell{1},'_',sprintf('%.0f',vStart),'-',sprintf('%.0f',vEnd),...
    '_',sprintf('%.2f',dv_calc),'_',sprintf('%.0f',SZA),'_',sprintf('%.0f',VZA),'_',fn_extra];

fnin = 'input_template.gc';
fnout = ['.',sfs,'inp',sfs,'input_',gas_cell{1},'_',sprintf('%.0f',vStart),'-',sprintf('%.0f',vEnd),...
    '_',sprintf('%.2f',dv_calc),'_',sprintf('%.0f',SZA),'_',sprintf('%.0f',VZA),'_',fn_extra,'.gc'];

fout = fopen(fnout,'w');
fin = fopen(fnin,'r');

while ~feof(fin)
    s = fgets(fin);
    s = strrep(s,'***fn_pre***',sprintf('%s',fn_pre));
    s = strrep(s,'***input_profile***',sprintf('%s',input_profile));
    
    s = strrep(s,'***SZA***',sprintf('%.1f',SZA));
    s = strrep(s,'***VZA***',sprintf('%.1f',VZA));
    
    s = strrep(s,'***vStart***',sprintf('%.1f',vStart));
    s = strrep(s,'***vEnd***',sprintf('%.1f',vEnd));
    
    s = strrep(s,'***dv_out***',sprintf('%f',dv_out));
    s = strrep(s,'***dv_calc***',sprintf('%f',dv_calc));
    s = strrep(s,'***FWHM***',sprintf('%f',FWHM));
    
    s = strrep(s,'***ngas***',sprintf('%.0f',ngas));
    s = strrep(s,'***gas_str***',sprintf('%s',gas_str));
    
    s = strrep(s,'***ReflSpectra***',sprintf('%s',ReflSpectra));
    
    s = strrep(s,'***K1***',sprintf('%f',K1));
    s = strrep(s,'***K2***',sprintf('%f',K2));
    s = strrep(s,'***K3***',sprintf('%f',K3));
    fprintf(fout,'%s',s);
end
fclose(fout);
fclose(fin);

% this command does not work if aerosols are added. Memory issues? maybe
% matlab has less memory resources than the shell terminal?
%unix(['./geocape_tool.exe ',fnout])
outp.fn = [fn_pre,'_upwelling_output.nc'];
outp.fnout = fnout;

function outp = F_write_GCtool_input(inp)
% Sadly unix(command) does not work in matlab when aerosols are added

% Written by Kang Sun 2017/09/05

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

K1 = inp.K1; K2 = inp.K2; K3 = inp.K3;

if isfield(inp,'fn_extra')
    fn_extra = inp.fn_extra;
else
    fn_extra = 'test';
end
fn_pre = [gas_cell{1},'_',sprintf('%.0f',vStart),'-',sprintf('%.0f',vEnd),...
    '_',sprintf('%.2f',dv_calc),'_',sprintf('%.0f',SZA),'_',sprintf('%.0f',VZA),'_',fn_extra,'_'];

if isfield(inp,'xsection_O2')
  xsection_O2 = inp.xsection_O2;
else
  xsection_O2 = ':   O2   -1     3 HITRAN';
end

fnin = 'input_template.gc';
fnout = ['input_',fn_pre,'.gc'];
fout = fopen(fnout,'w');
fin = fopen(fnin,'r');

while ~feof(fin)
    s = fgets(fin);
    s = strrep(s,'***fn_pre***',sprintf('%s',fn_pre));
    
    s = strrep(s,'***SZA***',sprintf('%.1f',SZA));
    s = strrep(s,'***VZA***',sprintf('%.1f',VZA));
    
    s = strrep(s,'***vStart***',sprintf('%.1f',vStart));
    s = strrep(s,'***vEnd***',sprintf('%.1f',vEnd));
    
    s = strrep(s,'***dv_out***',sprintf('%f',dv_out));
    s = strrep(s,'***dv_calc***',sprintf('%f',dv_calc));
    s = strrep(s,'***FWHM***',sprintf('%f',FWHM));

    s = strrep(s,'***ngas***',sprintf('%.0f',ngas));
    s = strrep(s,'***gas_str***',sprintf('%s',gas_str));
    
    s = strrep(s,'***K1***',sprintf('%f',K1));
    s = strrep(s,'***K2***',sprintf('%f',K2));
    s = strrep(s,'***K3***',sprintf('%f',K3));
    s = strrep(s,'***xsection_O2***',sprintf('%s',xsection_O2));
    fprintf(fout,'%s',s);
end
fclose('all');

%unix(['./geocape_tool.exe ',fnout])
outp.fn = [fn_pre,'upwelling_output.nc'];

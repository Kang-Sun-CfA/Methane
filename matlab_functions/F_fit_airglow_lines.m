function s2 = F_fit_airglow_lines(coeff,inp)

lines = inp.lines;
VQ = nan(length(lines.transitionWavenumber),1);
LQ = cell(length(lines.transitionWavenumber),5);
for i = 1:length(lines.transitionWavenumber)
    tmp = textscan(lines.lowerVibrationalQuanta(i,:),'%s%f','delimiter',' ',...
        'multipledelimsasone',1);
    VQ(i) = tmp{2};
    LQ{i,1} = lines.lowerLocalQuanta(i,2);
    LQ{i,2} = lines.lowerLocalQuanta(i,4:5);
    LQ{i,3} = lines.lowerLocalQuanta(i,6);
    LQ{i,4} = lines.lowerLocalQuanta(i,8:9);
    LQ{i,5} = lines.lowerLocalQuanta(i,end);
end
intqq = strcmp(LQ(:,1),'Q') & strcmp(LQ(:,3),'Q');
intpp = strcmp(LQ(:,1),'P') & strcmp(LQ(:,3),'P');
intrr = strcmp(LQ(:,1),'R') & strcmp(LQ(:,3),'R');

intpq = strcmp(LQ(:,1),'P') & strcmp(LQ(:,3),'Q');
intqp = strcmp(LQ(:,1),'Q') & strcmp(LQ(:,3),'P');

intrq = strcmp(LQ(:,1),'R') & strcmp(LQ(:,3),'Q');
intqr = strcmp(LQ(:,1),'Q') & strcmp(LQ(:,3),'R');

intsr = strcmp(LQ(:,1),'S') & strcmp(LQ(:,3),'R');
intrs = strcmp(LQ(:,1),'R') & strcmp(LQ(:,3),'S');

intop = strcmp(LQ(:,1),'O') & strcmp(LQ(:,3),'P');
intpo = strcmp(LQ(:,1),'P') & strcmp(LQ(:,3),'o');

int1 = VQ == 1;
int0 = VQ == 0;
intd = strcmp(LQ(:,5),'d');
intq = strcmp(LQ(:,5),'q');

inp.ints = {intrr|intrq,intpp|intpq};
inp.scale_ints = coeff(1:2);

outp = F_O21D_hitran(inp);
xx = double(inp.xx);
s2 = interp1(outp.wgrid,outp.xsec,xx,'linear','extrap');
s2 = double(s2(:)/max(s2)*coeff(3));
return
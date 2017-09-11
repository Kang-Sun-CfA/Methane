function outp = F_wrapper(inp)
% wrapper function to consolidate F_degrade_gc_output and F_gas_sens,
% calculate retrieval parameters (dofs, errors), and plot some results

% written by Kang Sun on 2017/09/11

outp = struct;

if ~isfield(inp,'fwhm')
    inp.fwhm  = nan(inp.nwin,1);
    for iwin = 1:inp.nwin
        inp.fwhm(iwin) = (inp.wmaxs(iwin)-inp.wmins(iwin))/inp.npixel(iwin)*inp.nsamp(iwin);
    end
end

% call function to subset and convolve gc output
outp_d = F_degrade_gc_output(inp);

% plot radiance, irradiance, and snr in the retrieval windows
outp.fig_overview = figure('unit','inch','position',[0 1 10 5],'color','w','visible','off');
nrow = 4;
for i = 1:2
    subplot(nrow,length(outp_d),i)
    plot(outp_d(i).wave,outp_d(i).rad,'linewidth',1,'color','k')
    set(gca,'linewidth',1,'xcolor','w','box','off','xlim',[min(outp_d(i).wave),max(outp_d(i).wave)])
    ylabel('Rradiance, I')
    title([num2str(inp.npixel(i)),'-pixel, FWHM = ',num2str(inp.fwhm(i),2),', d\lambda = ',num2str(inp.fwhm(i)/inp.nsamp(i),2)])
    
    subplot(nrow,length(outp_d),i+2)
    plot(outp_d(i).wave,outp_d(i).irrad,'linewidth',1,'color','k')
    set(gca,'linewidth',1,'xcolor','w','box','off','xlim',[min(outp_d(i).wave),max(outp_d(i).wave)])
    ylabel('Irradiance, F')
    title(['Spectral range: ',num2str(inp.wmins(i),'%.0f'),'-',num2str(inp.wmaxs(i),'%.0f'),' nm'])
    
    subplot(nrow,length(outp_d),i+4)
    plot(outp_d(i).wave,outp_d(i).rad./outp_d(i).irrad,'linewidth',1,'color','k')
    set(gca,'linewidth',1,'xcolor','w','box','off','xlim',[min(outp_d(i).wave),max(outp_d(i).wave)])
    ylabel('Normailzed radiance, I/F')
    title(['Mean albedo = ',num2str(mean(outp_d(i).surfalb))])
    
    subplot(nrow,length(outp_d),i+6)
    plot(outp_d(i).wave,outp_d(i).wsnr,'linewidth',1,'color','k')
    set(gca,'linewidth',1,'box','off','xlim',[min(outp_d(i).wave),max(outp_d(i).wave)])
%     Xtick = get(gca,'xtick');
%     Xtick = [min(outp_d(i).wave),Xtick(:)',max(outp_d(i).wave)];
%     set(gca,'xtick',unique(Xtick))
    xlabel('Wavelength [nm]')
    ylabel('SNR')
    title(['SNR_e = ',num2str(inp.snre(i))])
    
end

% call function to concatenate all windows and perform linear retrieval
outp_s = F_gas_sens(inp,outp_d);

% plot all weighting functions (jacobians)
outp.fig_jac = figure('unit','inch','position',[0 0 10 10],'color','w','visible','off');
xx = 1:outp_s.ny;
yy = outp_s.ywf;
for iv = 1:outp_s.nv
    yy(:,iv) = outp_s.ywf(:,iv)-mean(outp_s.ywf(:,iv));
    yy(:,iv) = yy(:,iv)/(max(yy(:,iv))-min(yy(:,iv)))+double(iv);
end
plot(xx,yy)
hold on
Ylim = [0,outp_s.nv+1];
plot([outp_s.nw(1),outp_s.nw(1)],Ylim,'--k','linewidth',1.5)
set(gca,'ylim',Ylim,'ytick',1:outp_s.nv,'yticklabel',outp_s.varnames,'xlim',[min(xx) max(xx)],'linewidth',1)
xlabel('Spectral index')
ylabel('State vector index')
hold off

% extract information from outp_s, the output of linear retrieval
ngas = length(outp_s.included_gases);
inc_prof = outp_s.inc_prof;
nz = outp_s.nz;
gasfidxs = outp_s.gasfidxs;
gastcol = outp_s.gastcol;
gascol = outp_s.gascol;
se = outp_s.se;
sa = outp_s.sa;
ak = outp_s.ak;
% gasnorm = outp_s.gasnorm;
% aircol = outp_s.aircol;

% relative vcd error for all gases
vcd_error = nan(ngas,1);
gas_dofs = nan(ngas,1);
for ig = 1:ngas
    if ig < ngas
        mat_interval = gasfidxs(ig):gasfidxs(ig+1)-1;
    
    else
        if inc_prof(ig)
            mat_interval = gasfidxs(ig):gasfidxs(ig)+nz-1;
        else
        	mat_interval = gasfidxs(ig);
        end
    end
    vcd_error(ig) = sqrt(sum(sum(se(mat_interval,mat_interval))))/gastcol(ig);
    gas_dofs(ig) = trace(ak(mat_interval,mat_interval));
end
% performance of profile retrievals
nprof = sum(inc_prof(1:ngas));
prof_dofs = nan(nprof);
prof_ak = nan(nz,nz,nprof);
% prof_ap = nan(nz,nprof);
prof_aperr = nan(nz,nprof);
prof_err = nan(nz,nprof);
if nprof > 0
    iprof = 0;
    for ig = 1:ngas
        if inc_prof(ig)
            iprof = iprof+1;
            mat_interval = gasfidxs(ig):gasfidxs(ig)+nz-1;
            prof_ak(:,:,iprof) = ak(mat_interval,mat_interval);
            prof_dofs(iprof) = trace(ak(mat_interval,mat_interval));
            prof_err(:,iprof) = sqrt(diag(se(mat_interval,mat_interval)))./gascol(:,ig);
            prof_aperr(:,iprof) = sqrt(diag(sa(mat_interval,mat_interval)))./gascol(:,ig);
        end
    end
end
outp.outp_s = outp_s;
outp.vcd_error = vcd_error;
outp.gas_dofs = gas_dofs;
outp.prof_ak = prof_ak;
outp.prof_dofs = prof_dofs;
outp.prof_aperr = prof_aperr;
outp.prof_err = prof_err;
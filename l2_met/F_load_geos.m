function geos = F_load_geos(inp)
% matlab function to load geos fp files. written by Kang Sun on 2019/04/15
end_datevec = inp.end_datevec;
start_datevec = inp.start_datevec;
start_datenum = datenum(start_datevec);
end_datenum = datenum(end_datevec);
geos_dir = inp.geos_dir;
if isfield(inp,'nlat')
    nlat = inp.nlat;
    nlon = inp.nlon;
    nlayer = inp.nlayer;
    step_hour = inp.step_hour;
else % default is geos fp
    step_hour = 3;
    nlat = 721;
    nlon = 1152;
    nlayer = 72;
end
nstep = (end_datenum-start_datenum)*24/step_hour+1;
fields2d = inp.fields2d;
fields_asm = inp.fields_asm;
if ~isfield(inp,'fields_chm')
    fields_chm = {'CO2','DELP'};
else
    fields_chm = inp.fields_chm;
end
if ~isfield(inp,'fields_aer')
    fields_aer = {};
else
    fields_aer = inp.fields_aer;
end
geos = [];
geos.nstep = nstep;
geos.nlat = nlat;
geos.nlon = nlon;
geos.nlat = nlat;
geos.nlayer = nlayer;
geos.datenum = nan(nstep,1);
geos.tai93 = nan(nstep,1);

for i = 1:length(fields2d)
    geos.(fields2d{i}) = nan(nlon,nlat,nstep,'single');
end
% extra 2d fields
geos.lapse_rate = nan(nlon,nlat,nstep,'single');
geos.xco2 = nan(nlon,nlat,nstep,'single');
for i = 1:length(fields_asm)
    geos.(fields_asm{i}) = nan(nlon,nlat,nlayer,nstep,'single');
end
for i = 1:length(fields_chm)
    geos.(fields_chm{i}) = nan(nlon,nlat,nlayer,nstep,'single');
end
for i = 1:length(fields_aer)
    geos.(fields_aer{i}) = nan(nlon,nlat,nlayer,nstep,'single');
end
if exist([geos_dir,'HS.mat'],'file')
    disp(['loading ',[geos_dir,'HS.mat']])
    tmp = load([geos_dir,'HS.mat']);
    geos.HS = tmp.HS;
else
fn = 'GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4';
disp(['loading ',[geos_dir,fn]])
a = F_ncread_selective([geos_dir,fn],{'PHIS'});
geos.HS = a.PHIS.data/9.8;
end
for istep = 1:nstep
    file_datenum = start_datenum+(istep-1)*step_hour/24;
    file_datevec = datevec(file_datenum);
    filedir = [geos_dir,'Y',num2str(file_datevec(1)),...
        '/M',num2str(file_datevec(2),'%02d'),'/D',num2str(file_datevec(3),'%02d'),'/'];
    % inst3_2d_asm_Nx
    fn = ['GEOS.fp.asm.inst3_2d_asm_Nx.',...
        datestr(file_datenum,'yyyymmdd_HH'),'00.V01.nc4'];
    disp(['loading ',filedir,fn])
    if ~isfield(geos,'lat')
        a = F_ncread_selective([filedir,fn],[{'lon','lat','TAITIME'},fields2d]);
        geos.lat = a.lat.data;
        geos.lon = a.lon.data;
    else
        a = F_ncread_selective([filedir,fn],[{'TAITIME'},fields2d]);
    end
    for field = fields2d
        geos.(char(field))(:,:,istep) = squeeze(a.(char(field)).data);
    end
    geos.datenum(istep) = file_datenum;
    geos.tai93(istep) = a.TAITIME.data;
    % inst3_3d_asm_Nv
    fn = ['GEOS.fp.asm.inst3_3d_asm_Nv.',...
        datestr(file_datenum,'yyyymmdd_HH'),'00.V01.nc4'];
    disp(['loading ',filedir,fn])
    a = F_ncread_selective([filedir,fn],fields_asm);
    for field = fields_asm
        geos.(char(field))(:,:,:,istep) = squeeze(a.(char(field)).data);
    end
    % lapse rate using first model layer
    geos.lapse_rate(:,:,istep) = -squeeze((geos.T(:,:,end,istep)-geos.T(:,:,end-1,istep))...
        ./(geos.H(:,:,end,istep)-geos.H(:,:,end-1,istep)));
    % inst3_3d_chm_Nv
    if ~isempty(fields_chm)
        fn = ['GEOS.fp.asm.inst3_3d_chm_Nv.',...
            datestr(file_datenum,'yyyymmdd_HH'),'00.V01.nc4'];
        disp(['loading ',filedir,fn])
        a = F_ncread_selective([filedir,fn],fields_chm);
        for field = fields_chm
            geos.(char(field))(:,:,:,istep) = squeeze(a.(char(field)).data);
        end
        % calculate xco2
        inp_g = [];
        inp_g.H = squeeze(geos.H(:,:,:,istep));
        variable_g = F_variable_g(inp_g);
        air_density_weight = squeeze(geos.DELP(:,:,:,istep)).*(1-squeeze(geos.QV(:,:,:,istep)))./variable_g;
        geos.xco2(:,:,istep) = sum(squeeze(geos.CO2(:,:,:,istep)).*air_density_weight,3)./ ...
            sum(air_density_weight,3);
    end
    % inst3_3d_aer_Nv
    if ~isempty(fields_aer)
        fn = ['GEOS.fp.asm.inst3_3d_aer_Nv.',...
            datestr(file_datenum,'yyyymmdd_HH'),'00.V01.nc4'];
        disp(['loading ',filedir,fn])
        a = F_ncread_selective([filedir,fn],fields_aer);
        for field = fields_aer
            geos.(char(field))(:,:,:,istep) = squeeze(a.(char(field)).data);
        end
    end
end
geos.T0 = squeeze(geos.T(:,:,end,:));
geos.fields2d = fields2d;
geos.fields_asm = fields_asm;
geos.fields_chm = fields_chm;
geos.fields_aer = fields_aer;
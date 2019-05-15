function variable = F_ncread_selective(filename,varname)
% matlab function to read netcdf file selectively, if you know which
% variables to load. these variables should be listed in cell array
% varname. written by Kang Sun on 2019/02/09

if isempty(varname)
    variable = [];
    return
end
ncid = netcdf.open(filename);

for ivar = 1:length(varname)
    varid = netcdf.inqVarID(ncid,varname{ivar});
    variable.(varname{ivar}).data = netcdf.getVar(ncid,varid);
    [~,~,~,natts] = netcdf.inqVar(ncid,varid);
    for k = 0:natts-1
        attname = netcdf.inqAttName(ncid,varid,k);
        fieldattname = ['A__',attname];
        if contains(fieldattname,' ')
            fieldattname = strrep(fieldattname,' ','_');
        end
        variable.(varname{ivar}).(fieldattname) = netcdf.getAtt(ncid,varid,attname);
    end
end

netcdf.close(ncid)
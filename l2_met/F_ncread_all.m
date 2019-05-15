function [variable, globalattribute] = F_ncread_all(filename)

ncid = netcdf.open(filename);
try
[ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid);
for i = 0:ngatts-1
    gattname = netcdf.inqAttName(ncid,netcdf.getConstant('NC_GLOBAL'),i);
    if strfind(gattname,'-')
        subgattname = strrep(gattname,'-','_');
    else
        subgattname = gattname;
    end
    globalattribute.(subgattname) = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'),gattname);
end
catch
    warning('Global attributes have problems!!');
    globalattribute = [];
end
for i = 0:nvars-1
    [varname,xtype,dimids,natts] = netcdf.inqVar(ncid,i);
   
    if isstrprop(varname(1), 'digit')
        varname = ['A__',varname];
    end
    variable.(varname).data = netcdf.getVar(ncid,i);
    for k = 0:natts-1
        attname = netcdf.inqAttName(ncid,i,k);
        fieldattname = ['A__',attname];
        variable.(varname).(fieldattname) = netcdf.getAtt(ncid,i,attname);
    end
end
netcdf.close(ncid)
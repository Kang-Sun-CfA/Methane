function tile = F_tile_retrieval(outp,inp)
% test stratified retrieval using many subpixels
% written by Kang Sun on 2019/05/20
if ~isfield(inp,'var_tile')
var_tile = {'a1w1','a2w1','a1w2','a2w2','sfcprs'};
else
    var_tile = inp.var_tile;
end
n_tile = inp.n_tile;
varnames = cat(1,outp.varnames(~ismember(outp.varnames,var_tile)),var_tile(:));
[~,locb] = ismember(varnames,outp.varnames);
ywf = outp.ywf(:,locb);
sa = outp.sa(locb,locb);
ny = outp.ny;
nv = outp.nv;
% imagesc(log10(outp.sa));colorbar
tile_varnames = cat(1,varnames,repmat(var_tile(:),[n_tile-1,1]));
for var = var_tile
    count = 1;
    for i = 1:length(tile_varnames)
        if strcmp(var{:},tile_varnames{i})
            tile_varnames{i} = [tile_varnames{i},'_tile',num2str(count)];
            count = count+1;
        end
    end
end
var_tile_idxs = [];
for i_tile = 1:n_tile
    tmp_var_tile = var_tile;
    for ivar = 1:length(var_tile)
        tmp_var_tile{ivar} = [tmp_var_tile{ivar},'_tile',num2str(i_tile)];
    end
var_tile_idxs.(['tile',num2str(i_tile)]) = find(ismember(tile_varnames,tmp_var_tile));
end

nvt = length(tile_varnames);
tile_ywf = zeros(ny*n_tile,nvt);

for i_tile = 1:n_tile
    row_idx = (1:ny)+(i_tile-1)*ny;
    tmp_varnames = varnames;
    for ivar = 1:outp.nv
        if ismember(tmp_varnames{ivar},var_tile)
            tmp_varnames{ivar} = [tmp_varnames{ivar},'_tile',num2str(i_tile)];
        end
    end
%     tmp_varnames
    col_idx = ismember(tile_varnames,tmp_varnames);
%     col_idx
    tile_ywf(row_idx,col_idx) = ywf;
end
% imagesc((outp.ywf));colorbar
tile_sa = zeros(nvt,nvt);
tile_sa(1:nv,1:nv) = sa;
for i_tile = 1:n_tile
    tmp_idx = var_tile_idxs.(['tile',num2str(i_tile)]);
    tile_sa(tmp_idx,tmp_idx) = sa(var_tile_idxs.tile1,var_tile_idxs.tile1);
end
tile_syn1 = diag(repmat(double(outp.wsnr./outp.rad),[n_tile,1]).^2);

san1 = inv(tile_sa);
temp = tile_ywf' * tile_syn1 * tile_ywf;
se = inv(temp + san1);
contri = se * tile_ywf' * tile_syn1;
ak = contri * tile_ywf;
sn  = (contri/(tile_syn1)) * (contri)';
ss  = (ak - eye(nvt)) * tile_sa * (ak - eye(nvt))';

h = outp.aircol/sum(outp.aircol);
h = h(:);
tile_k = zeros(size(ak,1),1);
tile_k(1:length(h)) = h;

xch4e_a = sqrt(tile_k'*tile_sa*tile_k);
xch4e_m = sqrt(tile_k'*sn*tile_k);
% xch4e_s = sqrt(h'*ss(gasfidxs(1):(gasfidxs(2)-1),gasfidxs(1):(gasfidxs(2)-1))*h);
xch4e_s = sqrt(tile_k'*ss*tile_k);
xch4e_r = sqrt(tile_k'*se*tile_k);
tile.k = tile_k;
tile.ss = ss;
tile.sa = tile_sa;
tile.sn = sn;
tile.se = se;
tile.ak = ak;
tile.varnames = tile_varnames;
tile.var_tile_idxs = var_tile_idxs;
tile.xch4e_a = xch4e_a;
tile.xch4e_s = xch4e_s;
tile.xch4e_m = xch4e_m;
tile.xch4e_r = xch4e_r;

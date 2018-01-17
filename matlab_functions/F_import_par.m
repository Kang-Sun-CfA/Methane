function lines = F_import_par(filename)
fileID = fopen(filename, 'r', 'native', 'US-ASCII');
fseek(fileID, 0, 'eof');
fileSizeInBytes = ftell(fileID);
fseek(fileID, 0, 'bof');
numberOfLines = floor(fileSizeInBytes/162);

lines.moleculeNumber = zeros(numberOfLines,1);
lines.isotopologueNumber = zeros(numberOfLines,1);
lines.transitionWavenumber = zeros(numberOfLines,1);
lines.lineIntensity = zeros(numberOfLines,1);
lines.einsteinACoefficient = zeros(numberOfLines,1);
lines.airBroadenedWidth = zeros(numberOfLines,1);
lines.selfBroadenedWidth = zeros(numberOfLines,1);
lines.lowerStateEnergy = zeros(numberOfLines,1);
lines.temperatureDependence = zeros(numberOfLines,1);
lines.pressureShift = zeros(numberOfLines,1);
lines.upperVibrationalQuanta = char( zeros(numberOfLines, 15) );
lines.lowerVibrationalQuanta = char( zeros(numberOfLines, 15) );
lines.upperLocalQuanta = char( zeros(numberOfLines, 15) );
lines.lowerLocalQuanta = char( zeros(numberOfLines, 15) );
lines.errorCodes = char( zeros(numberOfLines, 6) );
lines.referenceCodes = char( zeros(numberOfLines, 12) );
lines.flagForLineMixing = char( zeros(numberOfLines, 1) );
lines.upperStatisticalWeight = zeros(numberOfLines, 1);
lines.lowerStatisticalWeight = zeros(numberOfLines, 1);

for n=1:numberOfLines
    lines.moleculeNumber(n,1) = str2double( fread(fileID, 2, 'char=>char') );
    lines.isotopologueNumber(n,1) = str2double( fread(fileID, 1, 'char=>char') );
    lines.transitionWavenumber(n,1) = str2double( fread(fileID, 12, 'char=>char') );
    lines.lineIntensity(n,1) = str2double( fread(fileID, 10, 'char=>char') );
    lines.einsteinACoefficient(n,1) = str2double( fread(fileID, 10, 'char=>char') );
    lines.airBroadenedWidth(n,1) = str2double( fread(fileID, 5, 'char=>char') );
    lines.selfBroadenedWidth(n,1) = str2double( fread(fileID, 5, 'char=>char') );
    lines.lowerStateEnergy(n,1) = str2double( fread(fileID, 10, 'char=>char') );
    lines.temperatureDependence(n,1) = str2double( fread(fileID, 4, 'char=>char') );
    lines.pressureShift(n,1) = str2double( fread(fileID, 8, 'char=>char') );
    lines.upperVibrationalQuanta(n,:) = fread(fileID, 15, 'char=>char');
    lines.lowerVibrationalQuanta(n,:) = fread(fileID, 15, 'char=>char');
    lines.upperLocalQuanta(n,:) = fread(fileID, 15, 'char=>char');
    lines.lowerLocalQuanta(n,:) = fread(fileID, 15, 'char=>char');
    lines.errorCodes(n,:) = fread(fileID, 6, 'char=>char');
    lines.referenceCodes(n,:) = fread(fileID, 12, 'char=>char');
    lines.flagForLineMixing(n,:) = fread(fileID, 1, 'char=>char');
    lines.upperStatisticalWeight(n,1) = str2double( fread(fileID, 7, 'char=>char') );
    lines.lowerStatisticalWeight(n,1) = str2double( fread(fileID, 7, 'char=>char') );
    if fread(fileID,1) ~= 13
        error('The line endings of the HITRAN par file should be CR+LF, however the CR was missing.');
    end
    if fread(fileID,1) ~= 10
        error('The line endings of the HITRAN par file should be CR+LF, however the LF was missing.');
    end
end
fclose(fileID);
end
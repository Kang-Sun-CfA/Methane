function [header, data, ts] = readSeqSciCam(filename)

% Read spectrometer frames from StreamPix .seq file

%% Read header
    
% open the file for binary read
fid = fopen(filename,'r','l');

% read image characteristics
fseek(fid,548, 'bof');
header.ImageWidth = fread(fid,1,'uint32');
header.ImageHeight = fread(fid,1,'uint32');
header.ImageBitDepth = fread(fid,1,'uint32');
header.ImageBitDepthTrue = fread(fid,1,'uint32');
header.ImageSizeBytes = fread(fid,1,'uint32');

% read number of frames
fseek(fid,572,'bof');
header.NumFrames = fread(fid,1,'uint16');

% read true image size
fseek(fid,580,'bof');
header.TrueImageSize = fread(fid,1,'uint32');

% close the file
fclose(fid);

%% Read data and timestamps

if nargout > 1
    
    nframes = header.NumFrames;
    npix = header.ImageHeight*header.ImageWidth;
    
    % open the file for binary read
	fid = fopen(filename,'r','l');
    
    % seek past the header
	fseek(fid,8192,'bof');
    
    % read the whole file into unsigned 16 bit integers
	A = fread(fid,inf,'uint16=>uint16');
    
    % close the file
    fclose(fid);
    
    % reshape to put frames in the second dimension
    A = reshape(A,[],nframes);
    
    % the first npix elements of each frame is the image
    data = reshape(A(1:npix,:),header.ImageWidth,header.ImageHeight,nframes);
    
    % the next 2 elements are the whole seconds since 1970, 
    % stored in a 32 bit unsigned integer
    ts_sec = double(A(npix+1,:))' + (2^16)*double(A(npix+2,:))';
    
    % the next element is the fractional part of the timestamp, 
    % in whole milliseconds
    ts_ms = double(A(npix+3,:))';

    % combine to form the Unix-style timestamp
    ts = (ts_sec + 0.001*ts_ms);
    
    % combine to form the Matlab-style timestamp
    % ts = (ts_sec + 0.001*ts_ms)/86400 + datenum(1970,1,1);
end


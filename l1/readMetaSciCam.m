function meta = readMetaSciCam(data,col)

% Read meta data from each spectrometer frame

for i = 1:size(data,3)
    % strip off the meta data column (uint16, like the data)
    temp = squeeze(data(:,col,i));
        
    % change the endianness, then typecast to uint8
    temp = typecast(swapbytes(temp),'uint8');
    
    % read each field
    partNum{i} = cellstr(char(temp(3:34)'));% part number
    serNum{i} = cellstr(char(temp(35:48)'));% serial number
    fpaType{i} = cellstr(char(temp(49:64)'));% focal plane type
    
    crc(i) = typecast(temp(68:-1:65),'uint32');% CRC
    
    frameCounter(i) = typecast(temp(72:-1:69),'int32');% frame counter
    frameTime(i) = typecast(temp(76:-1:73),'single');% frame time (s)
    intTime(i) = typecast(temp(80:-1:77),'single');% integration time (s)
    
    freq(i) = typecast(temp(84:-1:81),'single'); % clock frequency (MHz)
    boardTemp(i) = typecast(temp(124:-1:121),'single'); % electronics temp (C)
    rawNUC(i) = typecast(temp(126:-1:125),'uint16'); % raw/NUC flag
    
    colOff(i) = typecast(temp(132:-1:131),'int16'); % column offset
    numCols(i) = typecast(temp(134:-1:133),'int16')+1; % number of columns
    rowOff(i) = typecast(temp(138:-1:137),'int16'); % row offset
    numRows(i) = typecast(temp(140:-1:139),'int16')+1; % number of rows
    
    % time since power-on
    yr(i) = typecast(temp(194:-1:193),'int16'); 
    dy(i) = typecast(temp(196:-1:195),'int16'); 
    hr(i) = typecast(temp(198:-1:197),'int16'); 
    mn(i) = typecast(temp(200:-1:199),'int16'); 
    sc(i) = typecast(temp(202:-1:201),'int16'); 
    ms(i) = typecast(temp(204:-1:203),'int16'); 
    microsec(i) = typecast(temp(206:-1:205),'int16'); 
    
    fpaTemp(i) = typecast(temp(480:-1:477),'single'); % focal plane temp (C) 
        
    intTimeTicks(i) = typecast(temp(146:-1:143),'int32'); % integration time clock cycles
end  

% put the results in a structure
meta.partNum = partNum;             % part number
meta.serNum = serNum;               % serial number
meta.fpaType = fpaType;             % focal plane type
meta.crc = crc;                     % CRC
meta.frameCounter = frameCounter;   % frame counter
meta.frameTime = frameTime;         % frame time (s)
meta.intTime = intTime;             % integration time (s)
meta.freq = freq;                   % clock frequency (MHz)
meta.fpaTemp = fpaTemp;             % focal plane temp (C) 
meta.boardTemp = boardTemp;         % electronics temp (C)
meta.rawNUC = rawNUC;               % raw/NUC flag
meta.rowOff = rowOff;               % row offset
meta.numRows = numRows;             % number of rows
meta.colOff = colOff;               % column offset
meta.numCols = numCols;             % number of columns
meta.year = yr;                     % time since power-on, whole years
meta.day = dy;                      % time since power-on, whole days
meta.hour = hr;                     % time since power-on, whole hours
meta.min = mn;                      % time since power-on, whole minutes
meta.sec = sc;                      % time since power-on, whole seconds
meta.millisec = ms;                 % time since power-on, whole millisec
meta.microsec = microsec;           % time since power-on, whole microsec
meta.intTimeTicks = intTimeTicks;   % integration time clock cycles

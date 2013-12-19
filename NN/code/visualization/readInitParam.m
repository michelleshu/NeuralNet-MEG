function InitParam = readInitParam(fname)

fprintf('Parameter file:\n')
which(fname)
%[Param,Tmp1,Value,Tm]=textread(fname,'%s%s%s','delimiter','=','commentstyle','matlab');
[Param,Tmp1,Value]=textread(fname,'%s%s%s','commentstyle','matlab');

for k=1:length(Param)
    ParamStr = deblank(char(lower(Param(k))));
    switch ParamStr   
        case 'sensorfile'
            InitParam.sensorfile = char(Value(k));
        case 'eogreject'
            InitParam.EOGreject = str2double(Value(k));
        case 'applyssp'
            InitParam.applySSP = str2double(Value(k));
        case 'applylnr'
            InitParam.applyLNR = str2double(Value(k));
        case 'freject'
            InitParam.Freject = str2double(Value(k));
        case 'dfdtreject'
            InitParam.DFDTreject = str2double(Value(k));
        case 'stimdelay'
            InitParam.stimDelay = str2double(Value(k));
    otherwise
       errorStr = strcat('Unknown parameter:   ',char(Param(k)));
       error(errorStr)
    end  

end


function P = plotEF(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20)
%
% --------------------------------------------------------------------------
% plotEF(f,EF1, EF2, ...)
%
% Plot evoked fields (EFs) for data generated by fiff2EF. 
% The plots are arranged according to their location on the 
% Neuromag-helmet (122 or 306 channels).
%
% timeVec = [1:N]        , time vector
%                          channels x time x graphs
% EF = [1:306,1:N]       , values for the 306 channels 
%                          channels x time x graphs
%  or
%
% EF = [1:122,1:N]       , values for the 122 channels 
%                          channels x time x graphs
% --------------------------------------------------------------------------
% plotEF('fifffile.fif', ...)
% 
% Plot evoked fields (EFs) for data in a FIFF-file, for instance generated
% by the Neuromag acquisition system.  
% --------------------------------------------------------------------------
% Optional Parameters & Values (in any order):
%
% 'ylimits'               = 'maxmin', scale to same data range for all coils. 
%                           'indi'   , scale to indivdual coils. 
%                           [ymin ymax], user defined for all coils
%                           {default = 'maxmin'}
% 'xlimits'               = 'maxmin', scale to data range for all coils.      
%                           [xmin xmax], user defined time range
%                           {default = 'maxmin'}
% 'coils'                 = 'grad1',  gradiometers, 1 
%                         = 'grad2',  gradiometers, 2 
%                         = 'grad12', gradiometers, aveerage 1 and 2 
%                         = 'mag',    magnetometers (only for Neuromag 306)   
%                            {default = 'grad1'}
% 'axes'                  = 'on', show axes with scales 
%                         = 'off', dont show axes         
%                           {default = 'on'}
% 'badchan'               = [ch1, ch2, ... , chN], list of bad chan.
%                           {default = []}
% 'labels'                = 'on', write coil names on each diagram
%                           'off'
%                           'numbers' write coil numbers on each diagram
%                           {default = 'on'}
% 'baseline'              = [tmin tmax], baseline for evoked fields.
%                           {default = no baseline subtracted} 
% 'SSP'                   = 'on' : apply SSP noise reduction - NB! only when displaying
%                                  FIFF-files
%                         = 'off' : don't apply SSP noise reduction 
%                           {default = 'off' }                             
% 'sets'                  = [set1, ..., setN], list of sets to be displayed
%                         = 'all', display all sets
%                           {default = 'all'}
% 'comment'               = 'string', write a string on top left corner
%                           use \n for line-break
%
%------------------------------------------------------------------------
% Ole Jensen, Brain Resarch Unit, Low Temperature Laboratory,
% Helsinki University of Technology, 02015 HUT, Finland,
% Report bugs to ojensen@neuro.hut.fi
%------------------------------------------------------------------------

%    Copyright (C) 2000 by Ole Jensen 
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You can find a copy of the GNU General Public License
%    along with this package (4DToolbox); if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


nargs = nargin;

   
InitParam = readInitParam('InitParam.txt');

if isstr(p1)
    % FIFF-file to be plottet
    j = 2;
    [Nset,Cset] = loadfif(p1,'sets');
    fprintf('Sets:\n')
    maxCol = 0;
    t0Col = 0;
    for k=1:Nset 
        fprintf('Set %d: %s\n',k,char(Cset(k))) 
        [Ptmp,Fs,t0]=loadfif(p1,k-1);
        if floor(abs(Fs*t0)) > abs(t0Col)      
            t0Col = floor(abs(Fs*t0)); 
        end
        if size(Ptmp,2) > maxCol
            maxCol = size(Ptmp,2);
        end
    end 
    P = zeros(size(Ptmp,1),maxCol,Nset); 
    for k=1:Nset 
        [Ptmp,Fs,t0]=loadfif(p1,k-1);
        P(:, t0Col - floor(abs(Fs*t0))+(1:size(Ptmp,2)),k) = Ptmp;

    end
    t = (-t0Col+(1:size(P,2)))/Fs;
    % Load SSP trans

else
    % Data matrix to be plottet
   SSP = 'off';
   SSPtrans = [];
   t = p1;
   P(:,:,1) = p2;
   j=3;
   while j <= nargs & ~isstr(eval(['p',int2str(j)]))                
      P(:,:,j-1) = eval(['p',int2str(j)]);
      j = j + 1;
   end 
end 



clf

fieldData = 0;
GridOn = 0;


m306 = 0;
m122 = 0; 


[fid,mess] = fopen(InitParam.sensorfile,'r');
if ~isempty(mess)
     tmpstr = strcat('Cannot open the file with coil definitions: ' ,InitParam.sensorfile);
     error(tmpstr);
end
fclose(fid);

[tmp,cx,cy,cl] = textread(InitParam.sensorfile,'%f %f %f %s');

if strcmp(InitParam.sensorfile,'loc122.txt')
    %m122 = 1;
    rowGrad = 1:122;
    rowGrad1 = 1:2:122;
    rowGrad2 = 2:2:122;
    rowMag = [];
    COILS = 'grad1'
elseif strcmp(InitParam.sensorfile,'loc306.txt') 
    rowGrad = [];
    for k=1:306
           if mod(k,3) ~= 0
               rowGrad = [rowGrad k];    
           end
    end
    rowGrad1 = 1:3:306;
    rowGrad2 = 2:3:306;
    rowMag = 3:3:306;
    %m306 = 1; 
    COILS = 'grad1';
else
    rowEEG = 1:length(cx);
    COILS = 'eeg';
end


fieldData = 0;
if (strcmp(InitParam.sensorfile,'loc122.txt') |  strcmp(InitParam.sensorfile,'loc306.txt'))  & max(max(max(P(rowGrad,:)))) < 1e-10
    fieldData = 1;
    P(rowGrad,:) = 1e13*P(rowGrad,:);
    P(rowMag,:) = 1e15*P(rowMag,:);
end

YLIMITS = 'maxmin';
XLIMITS = 'default';
AXES    = 'on';
LABELS  = 'on';
LOG     = 'off';
BADCH   = [];
COMMENT = '';
BASELINE = [];
SETS = 'all';
SSP = 'off';
if nargs > 1
  for i = j:2:nargs
    Param = eval(['p',int2str(i)]);
    Value = eval(['p',int2str(i+1)]);
    if ~isstr(Param)
      error('plotER(): Parameter must be a string')
    end
    Param = lower(Param);
    switch lower(Param)
       case 'xlimits'
         XLIMITS = Value;
       case 'ylimits'
         YLIMITS = Value;
       case 'coils'
         COILS   = Value;
       case 'axes'
         AXES = Value;
       case 'badchan'
         BADCH = Value;
       case 'labels'
         LABELS = Value;
       case 'baseline'
         BASELINE = Value;
       case 'comment'	 
         COMMENT = Value;
       case 'ssp'	 
         SSP = Value;
       case 'sets'	 
         SETS= Value;
    otherwise
      error('Unknown parameter.')
    end
  end
end

if strcmp(SSP,'on')
    megmodel([0 0 0],p1);
    [badchanlist,badchannames] = badchans;
    SSPtrans = projmat(p1,badchannames);
else 
    SSPtrans = [];
end



if ~isempty(BASELINE)
    tidx = find(t >= BASELINE(1) & t <= BASELINE(2)); 

    for k=1:size(P,3) 
        Ptmp = P(:,:,k);
        tmp = mean(Ptmp(:,tidx),2);
	
        for l=1:size(P,1)
            Ptmp(l,:) = Ptmp(l,:) - tmp(l);
        end
        P(:,:,k) = Ptmp; 
    end
end

amin = NaN;
amax = NaN;

if ~strcmp(SETS,'all')
   P = P(:,:,SETS); 
end
if ~isempty(SSPtrans) & strcmp(SSP,'on')
   for k=1:size(P,3)
        P(:,:,k) = SSPtrans*P(:,:,k);
   end
end

 
for j=1:length(BADCH)
    P(BADCH(j),:) = NaN;
end

YLBL = 'fT/cm';
if isstr(COILS)
    if strcmp(COILS,'grad1')
        P = P(rowGrad1,:,:);
        nplts = length(rowGrad1);
        cname = 'Gradiometers 1';
    elseif strcmp(COILS,'grad2')
        P = P(rowGrad2,:,:);
        nplts = length(rowGrad2);
        cname = 'Gradiometers 2';  
    elseif strcmp(COILS,'grad12')
        P = (P(rowGrad1,:,:)+P(rowGrad2,:,:))/2;
        nplts = length(rowGrad1);
        cname = 'Gradiometers 1 and 2';
    elseif strcmp(COILS,'mag')
        if m122
            error('This option is only allow for Neuromag306')
        end
        P = P(rowMag,:,:);
        nplts = length(rowMag);
        cname = 'Magnetometers';
        YLBL = 'fT';
    elseif strcmp(COILS,'eeg')
        nplts = length(rowEEG);
        cname = 'EEG'
    else
       error('Parameters coils must be  grad1, grad2, grad12, mag or eeg ')
    end
end

xmin = 2;
xmax = 14;
if isstr(XLIMITS)
    if strcmp(XLIMITS,'default')
        xmin = min(t);
        xmax = max(t);   
    end
else
    xmin = XLIMITS(1);
    xmax = XLIMITS(2);
end

xind = find(t >= xmin & t <= xmax);
P = P(:,xind,:);
t = t(xind); 

if isstr(YLIMITS)
    if strcmp(YLIMITS,'maxmin')
      amin = min(min(min(P)));
      amax = max(max(max(P)));
    end
else
    amin = YLIMITS(1);
    amax = YLIMITS(2);
end
px = 0.05+0.95*(cx - min(cx))/(0.06 + max(cx) - min(cx));
py = 0.9*(cy  - min(cy))/(0.04 + max(cy) - min(cy));
py =  py + 0.03;
px = 0.9*px;
for pos=1:nplts          
    fprintf('%d ',pos); 
    
    if strcmp(InitParam.sensorfile,'loc122.txt')
        subplotOL('position',[px(rowGrad1(pos)) py(rowGrad1(pos)) 0.060 0.060]);
    elseif strcmp(InitParam.sensorfile,'loc306.txt')
        subplotOL('position',[px(rowGrad1(pos)) py(rowGrad1(pos)) 0.058 0.053]);
    else
        subplotOL('position',[px(rowEEG(pos)) py(rowEEG(pos)) 0.1 0.1]);
    end
    
    if isnan(P(pos,1,:)) 
        plot([min(t) max(t)],[0 1],'r-')
        hold on
        plot([min(t) max(t)],[1 0],'r-')
        hold off
        set(gca,'YTickLabel',[])
        if strcmp(AXES,'off')
            axis off
        end 
    else 
        plot(t,squeeze(P(pos,:,:)));
        if ~isnan(amin)
            set(gca,'YLim',[amin amax]);
        end
        if ~isnan(amin)
            set(gca,'YTickLabel',[])
	end
	if GridOn 
            grid on
	end
        if strcmp(AXES,'off')
            axis off
        end 
    end


    set(gca,'FontSize',6);
    if strcmp(lower(LABELS),'numbers')  
        tmpstr = strcat(num2str(rowGrad1(pos)),'/',num2str(rowGrad2(pos)));
        title(tmpstr,'Color',[0 0 1]);
    end

    
    if strcmp(lower(LABELS),'numbers')
        if strcmp(COILS,'grad1')
            strtmp = num2str(rowGrad1(pos));
        elseif strcmp(COILS,'grad2')
            strtmp = num2str(rowGrad2(pos));
        elseif strcmp(COILS,'grad12')
            strtmp = strcat(num2str(rowGrad1(pos)),'/',num2str(rowGrad2(pos)));
        elseif strcmp(COILS,'mag')
            strtmp = num2str(rowMag(pos));
        else
            strtmp = numstr(rowEEG(pos));
        end
        title(strtmp,'Color',[0 0 1]);
    end

    
    if strcmp(lower(LABELS),'on')
        if strcmp(COILS,'grad1')
            strtmp = strcat('MEG',cl(rowGrad1(pos)));
        elseif strcmp(COILS,'grad2')
            strtmp = strcat('MEG',cl(rowGrad2(pos)));
        elseif strcmp(COILS,'grad12')
            strtmp = strcat('MEG',cl(rowGrad1(pos)),'/MEG',cl(rowGrad2(pos)));
        elseif strcmp(COILS,'mag')
            strtmp = strcat('MEG',cl(rowMag(pos)));
        else
            strtmp = cl(rowEEG(pos));
        end
        title(strtmp,'Color',[0 0 1]);
    end
  
    
    set(gca,'XLim',[min(t) max(t)]);
    set(gca,'XTickLabel',[])

end 



subplotOL('position',[0.8 0.9 0.058 0.053 ])

plot(t,zeros(size(t)))
set(gca,'XLim',[min(t) max(t)]);
% set(gca,'XLim',[xmin xmax]);
if ~isnan(amin)
    set(gca,'YLim',[amin amax]);
end
if GridOn
    grid on
end
% xlabel('Time (s)','FontSize',5)
if fieldData
    ylabel(YLBL,'FontSize',5)             
end   
set(gca,'FontSize',6);

if ~isempty(COMMENT)
   subplotOL('position',[0.1 0.9 0.1 0.05]);
   set(gca,'visible','off');
   th = text(0,1,sprintf(COMMENT));
   set(th,'fontsize',6);
end;

fprintf('\nYlimits: %d to %d \n',amin,amax)
   
suptitle(cname,15)
orient landscape


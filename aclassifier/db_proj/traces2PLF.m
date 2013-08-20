function [PLF,timeVec,freqVec] = traces2PLF(S,freqVec,Fs,width);
% function [PLF,timeVec,freqVec] = TFplf(S,freqVec,Fs,width);
%
% Calculates the phase locking factor for multiple trials using        
% multiple trials by applying the Morlet wavelet method.                            
%
% Input
% -----
% S    : signals = time x trials
% f    : frequencies over which to calculate spectrogram 
% Fs   : sampling frequency
% width: number of cycles in wavelet (> 5 advisable)  
%
% Output
% ------
% timeVec    : time
% freqVec    : frequency
% PLF    : phase-locking factor = frequency x time
%
%
% Ole Jensen, August 1998

S = S';
timeVec = (1:size(S,2))/Fs;  

B = zeros(length(freqVec),size(S,2)); 

for i=1:size(S,1)          
%    fprintf(1,'%d ',i); 
    for j=1:length(freqVec)
        B(j,:) = phasevec(freqVec(j),detrend(S(i,:)),Fs,width) + B(j,:);
    end
end
% fprintf('\n'); 
B = B/size(S,1);     


PLF = B;

function y = phasevec(f,s,Fs,width)
% function y = phasevec(f,s,Fs,width)
%
% Return a the phase as a function of time for frequency f. 
% The phase is calculated using Morlet's wavelets. 
%
% Fs: sampling frequency
% width : width of Morlet wavelet (>= 5 suggested).
%
% Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)


dt = 1/Fs;
sf = f/width;
st = 1/(2*pi*sf);

t=-3.5*st:dt:3.5*st;
m = morlet(f,t,width);

y = conv(s,m);

l = find(abs(y) == 0); 
y(l) = 1;

y = y./abs(y);
y(l) = 0;
   
y = y(ceil(length(m)/2):length(y)-floor(length(m)/2));



function y = morlet(f,t,width)
% function y = morlet(f,t,width)
% 
% Morlet's wavelet for frequency f and time t. 
% The wavelet will be normalized so the total energy is 1.
% width defines the ``width'' of the wavelet. 
% A value >= 5 is suggested.
%
% Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)
%
%
% Ole Jensen, August 1998 

sf = f/width;
st = 1/(2*pi*sf);
A = 1/sqrt(st*sqrt(pi));
y = A*exp(-t.^2/(2*st^2)).*exp(i*2*pi*f.*t);

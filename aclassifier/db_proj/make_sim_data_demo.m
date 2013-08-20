%% Test on Morelet and on Haar (Example 0)
t = linspace(0,1,1024);
pi2 = 2*pi;

s1_t = t*20-10;
sig2 = exp(-s1_t.^2/2) .* cos(5*s1_t);

sig1 = zeros(size(t));

sig1(257:513-128) = -1;
sig1((512-127):512) = 1;



make_plots(sig1,sig2,t,60,1:100,10,1000,'~/haar_morlet');




%% Example 1
% two sensors in phase in only one frequency band

time = linspace(0,1,1024);
sig1 = [ sin(32*pi2*time) ];
       % sin(20*pi2*time(701:1000)) + sin(36*pi2*time(701:1000)) + sin(16*pi2*time(701:1000))] ;

sig2 = [sin(64*pi2*(time)) +     sin(32*pi2*time)];
        %sin(12*pi2*(0.55+time(501:1000))) + sin(36*pi2*(time(501:1000)))];




make_plots(sig1,sig2,time,80,1:36,10,1024,'~/sample_one_band_in_phase');




%% Example 2
% two sensors in lagged phase in only one frequency band

time = linspace(0,1,1024);

sig1 = [sin(32*pi2*time) ];
sig2 = [sin(64*pi2*(0.2+time)) +     sin(32*pi2*(pi+time))];

make_plots(sig1,sig2,time,80,1:36,10,1024,'~/sample_one_band_lag_phase');


%% Example 3

sig1 = zeros(1,1024);
sig2 = zeros(1,1024);

sig1([100,400]) = 1;
sig2([400, 900]) = 1;

sig1([700:750]) = 0.7;
sig2([700:750]) = 0.7;

make_plots(sig1,sig2,time,60,1:36,10,1024,'~/sample_sharp');

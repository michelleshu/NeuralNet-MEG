function [ output_args ] = make_plots( sig1,sig2,time, f_thresh, scales, d_scale, freq, save_prefix )
% Create plots as seen in the report


[s1,f1,t1,p1]=spectrogram(sig1,128,0,1:f_thresh,freq);
[s2,f2,t2,p2]=spectrogram(sig2,128,0,1:f_thresh,freq);

m = min(p1(p1(:)>0));
p1(p1(:)==0) = m*10^-1;

m = min(p2(p2(:)>0));
p2(p2(:)==0) = m*10^-1;

f_cut = find(f1>=f_thresh,1);

% The raw signals 
clf;
colormap('jet');
subplot(2,1,1);
plot(time,sig1,'linewidth',1.5);
axis([min(time) max(time) min(sig1)-0.1 max(sig1)+0.1]);
xlabel('Time');
ylabel('Signal (arbitrary units)');
title('Signal 1');
%hold all;
subplot(2,1,2);
plot(time,sig2,'linewidth',1.5);
axis([min(time) max(time) min(sig2)-0.1 max(sig2)+0.1]);
%hold off;

xlabel('Time');
ylabel('Signal (arbitrary units)');
title('Signal 2');
save2pdf(sprintf('%s_signals.pdf',save_prefix));

%% spectrograms 

subplot(1,2,1);
surf(t1,f1(1:f_cut),log10(abs(p1(1:f_cut,:))),'edgecolor','none');
axis tight;
view(0,90);
set(gcf,'Renderer','Zbuffer')
cb = colorbar;
set(get(cb,'ylabel'),'String', 'Power (log 10)');
xlabel('Time');
ylabel('Frequency');
title('Spectrogram signal 1')

subplot(1,2,2);
surf(t2,f2(1:f_cut),log10(abs(p2(1:f_cut,:))),'edgecolor','none');
axis tight;
view(0,90);
set(gcf,'Renderer','Zbuffer')
cb = colorbar;
set(get(cb,'ylabel'),'String', 'Power (log 10)');
xlabel('Time');
ylabel('Frequency');
title('Spectrogram signal 2')

save2pdf(sprintf('%s_spectrogram.pdf',save_prefix));

%% Difference in power and phase

subplot(1,2,1);
surf(t1,f1(1:f_cut),(((angle(s1(1:f_cut,:)))-(angle(s2(1:f_cut,:))))),'edgecolor','none');
axis tight;
view(0,90);
set(gcf,'Renderer','Zbuffer')
colorbar;
xlabel('Time');
ylabel('Frequency');
title('Difference in Angle (STFT)')

subplot(1,2,2);
surf(t2,f2(1:f_cut),abs(abs(s1(1:f_cut,:)) - abs(s2(1:f_cut,:))),'edgecolor','none');
axis tight;
view(0,90);
set(gcf,'Renderer','Zbuffer')
colorbar;
xlabel('Time');
ylabel('Frequency');
title('Difference in Modulus (STFT)')

save2pdf(sprintf('%s_angle_mag_stft.pdf',save_prefix));

%% Morlet CWT
c_old = colormap();
c = colormap('bone');
c = c(end:-1:1,:);

colormap(c);
c1 = cwt(sig1,scales,'morl');
c2 = cwt(sig2,scales,'morl');


subplot(1,2,1)
surf(time, scales, abs(c1),'EdgeColor','none');
view(0,90);
set(gcf,'Renderer','Zbuffer')
axis tight;
xlabel('Time')
ylabel('Scale');
%colorbar;
title('Signal 1, absolute cwt coef (Morlet)')
colorbar()
colormap(c);

subplot(1,2,2)
surf(time, scales, abs(c2),'EdgeColor','none');
view(0,90);
set(gcf,'Renderer','Zbuffer')
axis tight;
xlabel('Time')
ylabel('Scale');
%colorbar;
title('Signal 2, absolute cwt coef (Morlet)');
colorbar()
colormap(c);

colormap(c);
save2pdf(sprintf('%s_cwt_coeff_morl.pdf',save_prefix));

%% PLV graph


subplot(1,1,1);
colormap(jet);
[p,t,f]=traces2PLF(sig1',1:f_thresh,freq,7);
[p2,t,f]=traces2PLF(sig2',1:f_thresh,freq,7);
plv = exp(sqrt(-1)*angle(p.*conj(p2)));
plot(abs(mean(plv,2)))


xlabel('Frequency (Hz)');
ylabel('PLV');
title('Phase Lock Value as a function of frequency for s1 and s2')
save2pdf(sprintf('%s_plv.pdf',save_prefix));


%% DWT and CWT for Haar
colormap(c);
sc = d_scale;
len=length(sig2);
[cf,l]=wavedec(sig2,sc,'haar');
% Compute and reshape DWT to compare with CWT.
cfd=zeros(sc,len);
for k=1:sc,
    d=detcoef(cf,l,k);
    d=d(ones(1,2^k),:);
    cfd(k,:)=wkeep(d(:)',len);
end
cfd=cfd(:);
I=find(abs(cfd) <sqrt(eps));
cfd(I)=zeros(size(I));
cfd=reshape(cfd,sc,len);

subplot(1,2,1); 

surf(time, 1:sc, abs(cfd),'EdgeColor','none');
set(gca,'yticklabel',wrev(1:sc))
view(0,90);
set(gcf,'Renderer','Zbuffer')
axis tight;
colormap(c);

%set(gca,'yticklabel',[]);
title('Sig 2: Discrete Transform, absolute coef');
ylabel('Level');
xlabel('Time')
colorbar();
% Compute CWT and compare with DWT
subplot(1,2,2);
ccfs=cwt(sig2,scales,'haar','plot');
title('Sig 2: Continuous Transform, absolute coef');
%set(gca,'yticklabel',[]);
ylabel('Scale');
xlabel('Time ticks')
colorbar();
colormap(c);
save2pdf(sprintf('%s_descr_haar_s_S2.pdf',save_prefix));


[cf,l]=wavedec(sig1,sc,'haar');
% Compute and reshape DWT to compare with CWT.
cfd=zeros(sc,len);
for k=1:sc,
    d=detcoef(cf,l,k);
    d=d(ones(1,2^k),:);
    cfd(k,:)=wkeep(d(:)',len);
end
cfd=cfd(:);
I=find(abs(cfd) <sqrt(eps));
cfd(I)=zeros(size(I));
cfd=reshape(cfd,sc,len);

subplot(1,2,1); 
%image(flipud(wcodemat(cfd,255,'m')));'
surf(time, 1:sc, abs(cfd),'EdgeColor','none');
set(gca,'yticklabel',wrev(1:sc))
view(0,90);
set(gcf,'Renderer','Zbuffer')
axis tight;

colormap(c);
%set(gca,'yticklabel',[]);
title('Sig 1: Discrete Transform, absolute coef');
ylabel('Level');
xlabel('Time')
colorbar();
% Compute CWT and compare with DWT
subplot(1,2,2);
ccfs=cwt(sig1,scales,'haar','plot');
title('Sig 1: Continuous Transform, absolute coef');
%set(gca,'yticklabel',scal2frq(scales,'haar',1/freq));
ylabel('Scale');
xlabel('Time ticks')
colorbar();
colormap(c);
save2pdf(sprintf('%s_descr_haar_s_S1.pdf',save_prefix));
colormap(c_old);

end


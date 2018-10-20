function [y,SNR]=symulacja_JW(fs,T,mu_imp1,f1_low,f1_high,s_noise,s_add)
% symulacja szum + cykliczne impulsy + artefakt
% T-czas w sekundach

% T=2.5; % sec
% fs=2^13; % Hz
ff1=5; % Hz
% ff2=17;
nx=fs*T; % samples
t=(1:nx)'/fs;

% impulsy to iid z rozkladu N(mu_imp,s_imp^2), a szum jest z rozkladu N(0,s_noise^2)
% mu_imp1=5;
% mu_imp2=8;
% s_noise=0.1;

rob1=randn(nx,1);
noise=rob1*s_noise;
impacts1=zeros(nx,1);
% impacts2=zeros(nx,1);

impact_ind1=100:round(fs/ff1):nx;
% impact_ind2=100:round(fs/ff2):nx;
impacts1(impact_ind1)=mu_imp1;
% impacts2(impact_ind2)=mu_imp2;

% impacts(impact_ind+1)=-0.5*impacts(impact_ind);
% impacts(impact_ind-1)=-0.5*impacts(impact_ind);

soi1=impacts1+noise;% signal of interest
% soi2=impacts2+noise;% signal of interest

% impacts(impact_ind+1)=-0.5*impacts(impact_ind);

dumping=200;
imp=exp(-t*dumping);

% f1_low=1500;f1_high=2500;
% f2_low=2000;f2_high=3500;

df=fs/nx;
win1=window(@hanning,round((f1_high-f1_low)/df));
% win2=window(@hanning,round((f2_high-f2_low)/df));

fft_mask1=(zeros(nx,1));
fft_mask1(round((f1_low)/df)+1:round((f1_high)/df))=win1;


% fft_mask2=(zeros(nx,1));
% 
% fft_mask2(round((f2_low)/df)+1:round((f2_high)/df))=win2;

fft_mask1=fft_mask1+rot90(fft_mask1,2);
% fft_mask2=fft_mask2+rot90(fft_mask2,2);

carrier1=real(ifft(fft(noise).*fft_mask1));
carrier1=carrier1./abs(hilbert(carrier1)); %???????????

% carrier2=real(ifft(fft(noise).*fft_mask2));
% carrier2=carrier2./abs(hilbert(carrier2)); %???????????
%carrier=sin(2*pi*1250*t);

response1=carrier1.*imp;
% response2=carrier2.*imp;


h1=response1; % impulse response of the system, 3 IFBs
% h2=response2;
nh=length(h1); % length of impulse response



y=conv(soi1,h1);%+conv(soi2,h2);
y=y(1:nx); y=y(:);

% additive noise
% s_add=0.7;
rob2=randn(nx,1);
n_add=rob2*s_add;

SNR=10*log10(sum(y.^2)/length(y)/(sum(n_add.^2)/length(n_add)));

y=y+n_add; % final signal

% y=y/max(abs(y));
end


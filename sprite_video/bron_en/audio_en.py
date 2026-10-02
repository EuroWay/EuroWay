import numpy as np, soundfile as sf, json, re
from numpy.fft import rfft, irfft
SR=44100; TL=json.load(open('timeline_en.json')); DUR=TL['end']; N=int(SR*DUR)
texts={}
for l in open('vo_en.txt',encoding='utf8'):
    if '|' in l: k,t=l.strip().split('|',1); texts[k]=t.replace('Bowsoft','Bouwsoft')
def conv(x,h):
    n=len(x)+len(h)-1; f=1<<(n-1).bit_length(); return irfft(rfft(x,f)*rfft(h,f),f)[:n]
rng=np.random.default_rng(1)
# small room impulse response
L=int(.35*SR); tt=np.arange(L)/SR; ir=rng.standard_normal(L)*np.exp(-tt/0.07); ir[0]=0; ir=ir/np.abs(ir).sum()*2.0
def voice_fx(a):
    # gentle low-mid warmth (+), tame harsh 4-7k, remove rumble
    F=rfft(a); fr=np.fft.rfftfreq(len(a),1/SR)
    g=np.ones_like(fr)
    g*= 1/(1+(70/np.maximum(fr,1))**4)
    g*= 1+0.35*np.exp(-((fr-220)/140)**2)
    g*= 1-0.30*np.exp(-((fr-5500)/1800)**2)
    g*= 1+0.15*np.exp(-((fr-2500)/800)**2)
    a=irfft(F*g,len(a))
    # soft compression
    env=np.convolve(np.abs(a),np.ones(441)/441,'same')+1e-4; thr=0.12
    gain=np.where(env>thr,(thr/env)**0.4,1.0); a=a*gain
    wet=conv(a,ir)[:len(a)]
    return a*0.88+wet*0.12
vo=np.zeros(N); caps=[]
for k,(st,en) in TL['vo'].items():
    a,sr=sf.read(f'voEN/{k}.wav')
    a=np.interp(np.arange(0,len(a)*SR/sr)*sr/SR,np.arange(len(a)),a)
    a=voice_fx(a); i=int(st*SR); vo[i:i+len(a)]+=a[:N-i]; d=len(a)/SR
    parts=[p.strip() for p in re.split(r'(?<=[.!?:])\s+',texts[k].replace('six p.m.','6 p.m.').replace('six percent','6%')) if p.strip()]; chunks=[]
    for p in parts:
        if len(p)>72 and ',' in p:
            cur=''
            for s_ in [s.strip() for s in p.split(',')]:
                if cur and len(cur)+len(s_)>60: chunks.append(cur+','); cur=s_
                else: cur=(cur+', '+s_) if cur else s_
            chunks.append(cur)
        else: chunks.append(p)
    tot=sum(len(c) for c in chunks); t=st
    for c in chunks: dd=d*len(c)/tot; caps.append([round(t,3),round(t+dd+.05,3),c]); t+=dd
json.dump(caps,open('caps.json','w'),ensure_ascii=False)
vo=vo/np.abs(vo).max()*0.9
t=np.arange(N)/SR; note=lambda m:440*2**((m-69)/12)
chords=[[57,60,64,67],[53,57,60,64],[48,52,55,59],[55,59,62,65]]; bar=60/100*4; mus=np.zeros(N)
for ci in range(int(DUR/bar)+1):
    ch=chords[ci%4]; s=int(ci*bar*SR); e=min(N,int((ci+1)*bar*SR)+int(.5*SR))
    if s>=N: break
    tt=np.arange(e-s)/SR; env=np.minimum(1,tt/0.8)*np.exp(-np.maximum(0,tt-bar)/0.4)
    for m in ch:
        f=note(m-12); mus[s:e]+=0.05*env*(np.sin(2*np.pi*f*tt)+0.3*np.sin(2*np.pi*2*f*tt+.3)+0.15*np.sin(2*np.pi*f*1.003*tt))
    for j in range(8):
        m=ch[[0,2,1,3,2,1,3,2][j]]+12; s2=s+int(j*bar/8*SR); Ln=min(int(.6*SR),N-s2)
        if Ln<=0: continue
        tt2=np.arange(Ln)/SR; mus[s2:s2+Ln]+=0.035*np.exp(-tt2*6)*np.sin(2*np.pi*note(m)*tt2)*(1-np.exp(-tt2*400))
    for j in range(4):
        s3=s+int(j*bar/4*SR); Ln=int(.25*SR)
        if s3+Ln>N: continue
        tt3=np.arange(Ln)/SR; mus[s3:s3+Ln]+=0.12*np.exp(-tt3*18)*np.sin(2*np.pi*(50+80*np.exp(-tt3*30))*tt3)
mus=np.convolve(mus,np.ones(6)/6,'same'); mus*=np.minimum(1,t/2)*np.minimum(1,(DUR-t)/3)
envv=np.convolve(np.abs(vo),np.ones(int(.25*SR))/int(.25*SR),'same'); mus=mus/np.abs(mus).max()*0.30*(1-0.55*np.clip(envv*12,0,1))
sfx=np.zeros(N)
def pop(ts,f=880,v=.16):
    i=int(ts*SR);Ln=int(.12*SR);tt=np.arange(Ln)/SR; sfx[i:i+Ln]+=v*np.exp(-tt*35)*np.sin(2*np.pi*(f+600*np.exp(-tt*50))*tt)
def whoosh(ts,v=.08):
    i=int(ts*SR);Ln=int(.5*SR);n=np.random.default_rng(int(ts*10)).standard_normal(Ln); n=np.convolve(n,np.ones(30)/30,'same'); sfx[i:i+Ln]+=v*n*np.sin(np.pi*np.arange(Ln)/Ln)**2
S={k:v[0] for k,v in TL['scenes'].items()}
for k,us in {'s1':[.3],'s2':[1.2,2.0,3.95,5.3,6.25,7.2,8.15,9.1,10.05],'s3':[.9,4.6],'s4':[.9,1.4,6.3],'s5':[.5,1.3,2.0,2.7,5.0],'s6':[.8,6.4,6.6],'s7':[4.2,5.9,7.9],'s8':[.6,4.9,5.0,8.3,8.7,9.1,9.5],'s9':[.3],'s10':[.3]}.items():
    for x in us: pop(S[k]+x)
def boom(ts,v=.35):
    i=int(ts*SR);Ln=int(.7*SR);tt=np.arange(Ln)/SR; sfx[i:i+Ln]+=v*np.exp(-tt*6)*np.sin(2*np.pi*(38+70*np.exp(-tt*12))*tt)
for k in S:
    if k!='s1': whoosh(S[k]-.05,.11)
boom(S['s7t']+.1); boom(S['s7']+.1,.22); boom(S['s10']+.05,.2)
whoosh(S['s7']+4.5,.1)
mix=vo+mus+sfx; mix=mix/np.abs(mix).max()*0.95
sf.write('mix_en.wav',np.stack([mix,mix],1),SR); print('ok',len(caps))

import numpy as np, soundfile as sf, json, re
SR=44100; DUR=82.0
N=int(SR*DUR)
starts={'s1':1.2,'s2':7.0,'s3':21.8,'s4':30.0,'s5':38.6,'s6':47.6,'s7':55.8,'s8':73.6}
texts=dict(l.strip().split('|',1) for l in open('vo.txt') if l.strip())
vo=np.zeros(N); caps=[]
for k,st in starts.items():
    a,sr=sf.read(f'vo/{k}.wav')
    if a.ndim>1:a=a.mean(1)
    # resample to SR
    a=np.interp(np.arange(0,len(a)*SR/sr)*sr/SR, np.arange(len(a)), a)
    i=int(st*SR); vo[i:i+len(a)]+=a[:N-i]
    d=len(a)/SR
    # caption chunks: split sentences, then long ones by comma
    parts=[p.strip() for p in re.split(r'(?<=[.!?:])\s+',texts[k]) if p.strip()]
    chunks=[]
    for p in parts:
        if len(p)>70 and ',' in p:
            sub=[s.strip() for s in p.split(',')]; cur=''
            for s_ in sub:
                if cur and len(cur)+len(s_)>60: chunks.append(cur+','); cur=s_
                else: cur=(cur+', '+s_) if cur else s_
            chunks.append(cur)
        else: chunks.append(p)
    tot=sum(len(c) for c in chunks); t=st
    for c in chunks:
        dd=d*len(c)/tot; caps.append([round(t,3),round(t+dd+.05,3),c]); t+=dd
json.dump(caps,open('caps.json','w'),ensure_ascii=False)
vo=vo/np.abs(vo).max()*0.9
# --- music: soft pad + pluck arpeggio, 100 bpm
t=np.arange(N)/SR
def note(m): return 440*2**((m-69)/12)
chords=[[57,60,64,67],[53,57,60,64],[48,52,55,59],[55,59,62,65]]  # Am7 Fmaj7 Cmaj7 G7
bar=60/100*4
mus=np.zeros(N)
for ci in range(int(DUR/bar)+1):
    ch=chords[ci%4]; s=int(ci*bar*SR); e=min(N,int((ci+1)*bar*SR)+int(.5*SR))
    if s>=N:break
    tt=np.arange(e-s)/SR
    env=np.minimum(1,tt/0.8)*np.exp(-np.maximum(0,tt-bar)/0.4)
    for m in ch:
        f=note(m-12)
        mus[s:e]+=0.05*env*(np.sin(2*np.pi*f*tt)+0.3*np.sin(2*np.pi*2*f*tt+.3)+0.15*np.sin(2*np.pi*f*1.003*tt))
    # arpeggio 8ths
    for j in range(8):
        m=ch[[0,2,1,3,2,1,3,2][j]]+12
        s2=s+int(j*bar/8*SR); L=int(.6*SR)
        if s2+L>N: L=N-s2
        if L<=0: continue
        tt2=np.arange(L)/SR
        mus[s2:s2+L]+=0.035*np.exp(-tt2*6)*np.sin(2*np.pi*note(m)*tt2)*(1-np.exp(-tt2*400))
    # soft kick on beats
    for j in range(4):
        s3=s+int(j*bar/4*SR); L=int(.25*SR)
        if s3+L>N: continue
        tt3=np.arange(L)/SR
        mus[s3:s3+L]+=0.12*np.exp(-tt3*18)*np.sin(2*np.pi*(50+80*np.exp(-tt3*30))*tt3)
# simple lowpass smoothing
k=np.ones(6)/6; mus=np.convolve(mus,k,'same')
mus*=np.minimum(1,t/2)*np.minimum(1,(DUR-t)/3)
# ducking under VO
envv=np.convolve(np.abs(vo),np.ones(int(.25*SR))/int(.25*SR),'same')
duck=1-0.55*np.clip(envv*12,0,1)
mus=mus/np.abs(mus).max()*0.32*duck
# SFX pops
sfx=np.zeros(N)
def pop(ts,f=880,v=.18):
    i=int(ts*SR);L=int(.12*SR);tt=np.arange(L)/SR
    sfx[i:i+L]+=v*np.exp(-tt*35)*np.sin(2*np.pi*(f+600*np.exp(-tt*50))*tt)
def whoosh(ts,v=.08):
    i=int(ts*SR);L=int(.5*SR);n=np.random.default_rng(int(ts*10)).standard_normal(L)
    n=np.convolve(n,np.ones(30)/30,'same');e=np.sin(np.pi*np.arange(L)/L)**2
    sfx[i:i+L]+=v*n*e
for ts in [0.4,8.25,9.25,11.15,22.3,27.0,30.7,31.3,35.4,39.0,44.6,48.4,53.9,54.1,55.9,60.6,60.8,66.2,66.75,67.3,67.85,74.0]: pop(ts)
for ts in [6.3,11.8,21.0,29.5,38.0,47.0,55.0,59.4,61.6,72.0]: whoosh(ts)
mix=vo+mus+sfx
mix=mix/np.abs(mix).max()*0.95
sf.write('mix.wav',np.stack([mix,mix],1),SR)
print('caps',len(caps))

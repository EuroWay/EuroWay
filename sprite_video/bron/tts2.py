import sys, re, json, numpy as np, soundfile as sf, onnxruntime as ort
from piper_phonemize import phonemize_espeak
OVR={'MAILBOX':'mˈeːlbɔks','SPRITE':'sprˈɑjt','VOLUME':'voːlˈyməkˌɔrtɪŋ','BOUWSOFT':'bˈʌusɔft'}
class Voice:
    def __init__(s,d,name):
        s.sess=ort.InferenceSession(f'{d}/{name}.onnx')
        s.tok={}
        for l in open(f'{d}/tokens.txt',encoding='utf8'):
            l=l.rstrip('\n'); 
            if not l: continue
            sym,i=l.rsplit(' ',1); s.tok[sym if sym else ' ']=int(i)
    def ids(s,ph):
        out=[s.tok['^'],s.tok['_']]
        for p in ph:
            if p in s.tok: out+=[s.tok[p],s.tok['_']]
        return out+[s.tok['$']]
    def say(s,text,length=1.0,noise=.75,nw=.9):
        # build phoneme string with overrides
        ph=''
        for part in re.split(r'(\[\[\w+\]\])',text):
            m=re.fullmatch(r'\[\[(\w+)\]\]',part)
            if m: ph+=OVR[m.group(1)]
            elif part:
                p=phonemize_espeak(part,'nl'); ph+=' '.join(''.join(x) for x in p)
        ids=np.array([s.ids(ph)],dtype=np.int64)
        a=s.sess.run(None,{'input':ids,'input_lengths':np.array([ids.shape[1]],dtype=np.int64),'scales':np.array([noise,length,nw],dtype=np.float32)})[0].squeeze()
        return a, ph
def render(voice,text,seed=0):
    rng=np.random.default_rng(seed)
    sents=[x for x in re.split(r'(?<=[.?!:])\s+',text.strip()) if x]
    out=[]; SR=22050
    for i,s_ in enumerate(sents):
        a,ph=voice.say(s_,length=0.95+rng.uniform(-.02,.03))
        # trim silence
        nz=np.where(np.abs(a)>0.01)[0]; a=a[max(0,nz[0]-200):nz[-1]+800] if len(nz) else a
        out.append(a)
        if i<len(sents)-1:
            pause=.32 if s_.endswith(('.','!','?')) else .18
            out.append(np.zeros(int(SR*pause)))
    return np.concatenate(out)
if __name__=='__main__':
    d,name,txtfile,outdir=sys.argv[1:5]
    v=Voice(d,name); import os; os.makedirs(outdir,exist_ok=True)
    for l in open(txtfile,encoding='utf8'):
        if '|' not in l: continue
        k,t=l.strip().split('|',1)
        a=render(v,t,seed=hash(k)%1000)
        sf.write(f'{outdir}/{k}.wav',a/np.abs(a).max()*.9,22050)
        print(k,round(len(a)/22050,2))

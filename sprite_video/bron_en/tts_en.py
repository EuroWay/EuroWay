import sys,os,re,numpy as np,soundfile as sf,sherpa_onnx
d='kokoro-multi-lang-v1_0'
cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(model=f'{d}/model.onnx',voices=f'{d}/voices.bin',tokens=f'{d}/tokens.txt',data_dir=f'{d}/espeak-ng-data',lexicon=f'{d}/lexicon-us-en.txt',lang='en-us'),num_threads=8),max_num_sentences=1)
tts=sherpa_onnx.OfflineTts(cfg); sid=int(sys.argv[1]); out=sys.argv[2]; speed=float(sys.argv[3]); os.makedirs(out,exist_ok=True)
for l in open('vo_en.txt'):
    if '|' not in l: continue
    k,t=l.strip().split('|',1)
    sents=[s for s in re.split(r'(?<=[.?!:])\s+',t) if s]; parts=[]
    for i,s in enumerate(sents):
        a=tts.generate(s,sid=sid,speed=speed); x=np.array(a.samples)
        nz=np.where(np.abs(x)>0.008)[0]; x=x[max(0,nz[0]-300):nz[-1]+1200]
        parts.append(x)
        if i<len(sents)-1: parts.append(np.zeros(int(a.sample_rate*(.28 if s.endswith(('.','?','!')) else .16))))
    y=np.concatenate(parts); sf.write(f'{out}/{k}.wav',y/np.abs(y).max()*.9,a.sample_rate); print(k,round(len(y)/a.sample_rate,2))

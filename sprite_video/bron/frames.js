const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const fs=require('fs');
const [W,NW,FPS]=[+process.argv[2],+process.argv[3],30];
const tl=JSON.parse(fs.readFileSync(__dirname+'/timeline.json','utf8')), caps=JSON.parse(fs.readFileSync(__dirname+'/caps.json','utf8'));
(async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1920,height:1080}});
  await p.goto('file://'+__dirname+'/site/index.html'); await p.evaluate(()=>document.fonts.ready);
  await p.evaluate(([a,c])=>setData(a,c),[tl,caps]);
  const total=Math.round(FPS*tl.end);
  for(let f=W;f<total;f+=NW){ await p.evaluate(t=>render(t),f/FPS); await p.screenshot({path:`frames/f${String(f).padStart(5,'0')}.jpg`,type:'jpeg',quality:92}); }
  await b.close();
})();

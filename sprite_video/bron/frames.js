const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const fs=require('fs');
const [W,NW,FPS,DUR]=[+process.argv[2],+process.argv[3],30,82];
const caps=JSON.parse(fs.readFileSync(__dirname+'/caps.json','utf8'));
(async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1920,height:1080}});
  await p.goto('file://'+__dirname+'/site/index.html'); await p.evaluate(()=>document.fonts.ready);
  await p.evaluate(c=>setCaps(c),caps);
  const total=Math.round(FPS*DUR);
  for(let f=W;f<total;f+=NW){
    await p.evaluate(t=>render(t),f/FPS);
    await p.screenshot({path:`frames/f${String(f).padStart(5,'0')}.jpg`,type:'jpeg',quality:92});
  }
  await b.close();
})();

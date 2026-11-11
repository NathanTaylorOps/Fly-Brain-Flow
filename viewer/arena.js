/* Arena Lab: rendering never steers agents. All positions and metrics come from the server. */
'use strict';
const $ = id => document.getElementById(id);
const canvas = $('arena'), ctx = canvas.getContext('2d'), chart = $('chart');
let state = null, env = null, running = true, pending = false, chain = Promise.resolve();
let zoom = 1, panX = 0, panY = 0, transform = {s:1,x:0,y:0}, W=800,H=550;
let selected = null, placement = false, trails = new Map(), particles = [], lastParticleTime=0;
let simBudget=0, lastTick=performance.now(), lastEnvironment=0, lastStateTime=0, activePreset='open';
let base = document.createElement('canvas'), baseDirty=true, hazardImage=document.createElement('canvas');
let hazardDirty=true, densityImage=document.createElement('canvas'), densityDirty=true;
let previousAgents=new Map(), renderAgents=new Map(), receivedAt=0, transitionMs=80;
let chartHistory=[], lastChartSecond=-1, requestSeconds=0, requestWall=0;
const colors={north:'#c4ee88',east:'#7ed4c2',south:'#f1b77d',west:'#aabfee'};
const fmt=(v,n=1)=>Number(v||0).toFixed(n);

async function api(path, body) {
 const r=await fetch('/api/'+path, body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 let d; try {d=await r.json();} catch {throw new Error('The simulation returned an unreadable response.');}
 if(!r.ok) throw new Error(d.error||'Simulation request failed.');
 return d;
}
function error(e){$('error').hidden=false;$('error').textContent=e.message||String(e);$('connection').textContent='Connection interrupted';$('connectionDot').style.background='#ff7866';}
function clearError(){$('error').hidden=true;$('connection').textContent='Local session connected';$('connectionDot').style.background='';}
function action(fn){chain=chain.then(async()=>{try {await fn();clearError();}catch(e){running=false;setPlay();error(e);}});return chain;}
async function environment(){env=await api('environment');hazardDirty=true;baseDirty=true;lastEnvironment=state?.time||0;}
function setPlay(){$('play').textContent=running?'Ⅱ Pause':'▶ Play';}
function accept(s, sync=false){
 const now=performance.now(); previousAgents=new Map(renderAgents.size?renderAgents:(state?.agents||[]).map(a=>[a.id,a]));
 transitionMs=Math.min(180,Math.max(30,now-receivedAt)); receivedAt=now;
 state=s; densityDirty=true;
 if(sync && s.alarm_time!==null && s.alarm_time!==undefined){chartHistory=[{t:0,n:0},...s.history.filter(v=>v.time>=s.alarm_time).map(v=>({t:v.time-s.alarm_time,n:v.evacuated}))];}
 for(const a of s.agents){let t=trails.get(a.id)||[];if(!t.length||Math.hypot(a.x-t[t.length-1][0],a.y-t[t.length-1][1])>.12)t.push([a.x,a.y]);if(t.length>90)t.shift();trails.set(a.id,t);}
 const liveIds=new Set(s.agents.map(a=>a.id));for(const id of trails.keys())if(!liveIds.has(id))trails.delete(id);
 if(s.time<lastStateTime){trails.clear();particles=[];chartHistory=[];lastChartSecond=-1;lastParticleTime=s.time;}
 const elapsed=s.metrics.evacuation_elapsed||0;
 if(s.phase!=='calm' && (Math.floor(elapsed*2)!==lastChartSecond || s.phase==='complete')){chartHistory.push({t:elapsed,n:s.evacuated});lastChartSecond=Math.floor(elapsed*2);if(chartHistory.length>1200)chartHistory.splice(1,1);}
 lastStateTime=s.time;
 updateNumbers();if(sync)syncControls();drawChart();
 if(s.phase==='complete'){running=false;setPlay();}
}
function syncControls(){
 if(!state)return; const c=state.config;
 $('population').value=c.population;$('populationValue').textContent=c.population+' flies';$('seed').value=c.seed;
 $('windDirection').value=c.wind_direction_deg;$('windSpeed').value=c.wind_speed_mps;$('hazardStrength').value=c.hazard_strength;
 updateWindLabels();
}
function updateWindLabels(){
 const d=Number($('windDirection').value), compass=['East','North-east','North','North-west','West','South-west','South','South-east'];
 $('windDirectionValue').textContent=compass[Math.round(d/45)%8]+' · '+d+'°';
 $('windSpeedValue').textContent=fmt($('windSpeed').value)+' m/s';$('windArrow').style.transform='rotate('+(-d)+'deg)';
 const h=Number($('hazardStrength').value);$('hazardValue').textContent=h?fmt(h)+'×':'Off';
}
function updateNumbers(){
 if(!state)return;const s=state,m=s.metrics,p=s.population,pct=100*s.evacuated/p;
 $('remaining').textContent=s.remaining;$('populationNote').textContent='of '+p+' agents';
 $('evacuated').innerHTML=Math.round(pct)+'<em>%</em>';$('evacuatedNote').textContent=s.evacuated+' reached an exit';$('evacProgress').style.width=pct+'%';
 $('flow').innerHTML=fmt(m.throughput_per_s)+'<em>/s</em>';$('density').innerHTML=fmt(m.peak_local_density??m.density,2)+'<em>/m²</em>';
 const mins=Math.floor(s.time/60),seconds=Math.floor(s.time%60),tenths=Math.floor((s.time%1)*10);
 $('clock').innerHTML=String(mins).padStart(2,'0')+':'+String(seconds).padStart(2,'0')+'<em>.'+tenths+'</em>';
 $('simSpeed').textContent=requestWall>0?fmt(requestSeconds/requestWall)+'× measured playback':'Simulation ready';
 const phase={calm:'Gathering',evacuating:'Evacuation in progress',complete:'Arena cleared'};
 $('phase').textContent=phase[s.phase]||s.phase;$('phase').classList.toggle('emergency',s.phase==='evacuating');
 $('alarm').textContent=s.phase==='calm'?'↗ Start evacuation':s.phase==='complete'?'✓ Arena cleared':'↗ Evacuation active';$('alarm').classList.toggle('active',s.phase!=='calm');$('alarm').disabled=s.phase!=='calm';
 $('curveTotal').textContent=s.evacuated+' / '+p;
 $('clearance').textContent=s.phase==='complete'?'Cleared in '+fmt(m.evacuation_elapsed)+' s':s.phase==='calm'?'Awaiting evacuation':fmt(m.evacuation_elapsed)+' s since alarm';
 $('overlap').textContent=fmt(100*(m.max_overlap||0),2)+' cm';$('stalled').textContent=m.stalled||0;
 updateExits();updateSelection();
}
function updateExits(){
 const parent=$('exitRows');
 if(parent.childElementCount!==state.exits.length){parent.replaceChildren();for(const e of state.exits){const b=document.createElement('button');b.className='exit-row';b.dataset.id=e.id;b.innerHTML='<span class="exit-name"></span><span class="exit-track"><i></i></span><span class="exit-count"></span><span class="exit-status"></span>';b.onclick=()=>action(async()=>{const current=state.exits.find(x=>x.id===e.id);accept(await api('control',{exit:{id:e.id,open:!current.open}}));await environment();});parent.append(b);}}
 for(const e of state.exits){const b=[...parent.children].find(x=>x.dataset.id===e.id),n=state.metrics.evacuated_by_exit?.[e.id]||0;b.classList.toggle('closed',!e.open);b.title=(e.open?'Close ':'Open ')+e.label;b.setAttribute('aria-label',(e.open?'Close ':'Open ')+e.label);b.querySelector('.exit-name').textContent=e.id[0].toUpperCase();b.querySelector('.exit-count').textContent=n;b.querySelector('.exit-status').textContent=e.open?'OPEN':'CLOSED';b.querySelector('.exit-track i').style.width=(100*n/state.population)+'%';b.querySelector('.exit-track i').style.background=colors[e.id];}
}
function updateSelection(){
 if(selected===null){$('selection').hidden=true;return;}const a=state.agents.find(a=>a.id===selected);$('selection').hidden=false;
 $('selectedId').textContent='FLY '+String(selected+1).padStart(3,'0');$('selectedState').textContent=a?({wandering:'Exploring',waiting:'Waiting',avoiding:'Avoiding repellent',evacuating:'Evacuating'}[a.state]||a.state):'Exited arena';
 $('selectedSpeed').textContent=a?fmt(a.speed,2)+' m/s':'—';$('selectedExit').textContent=a?.exit_id?capital(a.exit_id):'—';$('selectedExposure').textContent=a?fmt(a.exposure,2):'—';
}
function capital(v){return v.charAt(0).toUpperCase()+v.slice(1);}
function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(2,devicePixelRatio||1);W=r.width;H=r.height;canvas.width=Math.round(W*d);canvas.height=Math.round(H*d);ctx.setTransform(d,0,0,d,0,0);base.width=canvas.width;base.height=canvas.height;baseDirty=true;drawChart();}
new ResizeObserver(resize).observe($('stage'));
function worldTransform(c){const w=env?.width||64,h=env?.height||48,s=Math.min((W-55)/w,(H-65)/h)*zoom;transform={s,x:(W-w*s)/2+panX,y:(H+h*s)/2+panY};c.translate(transform.x,transform.y);c.scale(s,-s);$('scale').style.width=(10*s)+'px';$('scale').textContent='10 m';$('scale').style.borderTop='1px solid #7f9a88';}
function textAt(c,t,x,y,size=9,color='#829d87',align='center'){c.save();c.translate(x,y);c.scale(1,-1);c.fillStyle=color;c.font=size/transform.s+'px "Segoe UI",sans-serif';c.textAlign=align;c.fillText(t,0,0);c.restore();}
function path(c,poly){c.beginPath();poly.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.closePath();}
function drawBase(){
 const d=canvas.width/W,b=base.getContext('2d');b.setTransform(d,0,0,d,0,0);b.clearRect(0,0,W,H);
 const g=b.createRadialGradient(W*.48,H*.48,10,W*.48,H*.48,W*.7);g.addColorStop(0,'#213126');g.addColorStop(1,'#101b18');b.fillStyle=g;b.fillRect(0,0,W,H);
 b.strokeStyle='#293b2c40';b.lineWidth=.5;for(let x=0;x<W;x+=24){b.beginPath();b.moveTo(x,0);b.lineTo(x,H);b.stroke();}for(let y=0;y<H;y+=24){b.beginPath();b.moveTo(0,y);b.lineTo(W,y);b.stroke();}
 b.save();worldTransform(b);
 if(env){
  // Geometry comes directly from the simulation. Decorative seating stays inside obstacles.
  for(const poly of env.outer){path(b,poly);b.fillStyle='#26382b';b.fill();}
  b.save();b.beginPath();for(const poly of env.outer){poly.forEach((p,i)=>i?b.lineTo(...p):b.moveTo(...p));b.closePath();}b.clip();
  b.strokeStyle='#a5c1900a';b.lineWidth=.08;for(let x=0;x<=64;x+=2){b.beginPath();b.moveTo(x,0);b.lineTo(x,48);b.stroke();}for(let y=0;y<=48;y+=2){b.beginPath();b.moveTo(0,y);b.lineTo(64,y);b.stroke();}
  // Event floor graphics do not add hidden collisions.
  b.fillStyle='#31442f';b.fillRect(23,12,18,18);b.strokeStyle='#a9c49324';b.lineWidth=.1;b.strokeRect(23.7,12.7,16.6,16.6);b.beginPath();b.arc(32,21,4,0,Math.PI*2);b.stroke();b.beginPath();b.moveTo(23.7,21);b.lineTo(40.3,21);b.stroke();textAt(b,'EVENT FLOOR',32,20.5,11,'#b3cb9d48');
  b.restore();
  const mask=env.walkable,nx=env.grid_width,ny=env.grid_height,r=env.resolution,[ox,oy]=env.origin;
  b.beginPath();for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){if(!mask[j*nx+i])continue;const x=ox+i*r,y=oy+j*r;if(i===0||!mask[j*nx+i-1]){b.moveTo(x,y);b.lineTo(x,y+r);}if(i===nx-1||!mask[j*nx+i+1]){b.moveTo(x+r,y);b.lineTo(x+r,y+r);}if(j===0||!mask[(j-1)*nx+i]){b.moveTo(x,y);b.lineTo(x+r,y);}if(j===ny-1||!mask[(j+1)*nx+i]){b.moveTo(x,y+r);b.lineTo(x+r,y+r);}}
  b.strokeStyle='#091710';b.lineWidth=.6;b.stroke();b.strokeStyle='#789164';b.lineWidth=.13;b.stroke();
  for(const o of env.obstacles){
   b.fillStyle='#0d1c15';b.fillRect(o.x+.3,o.y-.4,o.width,o.height);b.fillStyle=o.id==='stage'?'#15271d':'#1b2d20';b.fillRect(o.x,o.y,o.width,o.height);b.strokeStyle='#658255';b.lineWidth=.1;b.strokeRect(o.x,o.y,o.width,o.height);
   if(o.id.startsWith('seats')){for(let j=.65;j<o.height-.3;j+=.65){b.fillStyle='#345039';b.fillRect(o.x+.35,o.y+j-.08,o.width-.7,.11);for(let i=.55;i<o.width-.3;i+=.55){b.fillStyle=((Math.floor(i*2)+Math.floor(j*2))%5===0)?'#5c7451':'#496441';b.fillRect(o.x+i,o.y+j,.32,.32);}}textAt(b,o.id.endsWith('nw')?'SECTOR A':o.id.endsWith('ne')?'SECTOR B':o.id.endsWith('sw')?'SECTOR D':'SECTOR C',o.x+o.width/2,o.y+o.height/2,8,'#d6e5bd80');}
   else if(o.id==='stage'){b.strokeStyle='#caed7d70';b.lineWidth=.14;b.strokeRect(o.x+.6,o.y+.6,o.width-1.2,o.height-1.2);textAt(b,'MAIN STAGE',o.x+o.width/2,o.y+2.2,10,'#bed9a2');for(let i=1;i<12;i+=2){b.fillStyle='#bad56a';b.fillRect(o.x+i,o.y+.2,.25,.15);}}
   else{textAt(b,'SERVICES',o.x+o.width/2,o.y+o.height/2,7,'#96b48d');}
  }
  textAt(b,'NORTH CONCOURSE',32,40.2,8,'#8ca47c');textAt(b,'SOUTH CONCOURSE',32,8.7,8,'#8ca47c');
 }
 b.restore();baseDirty=false;
}
function gridIndex(x,y,h=env?.hazard){if(!h)return-1;const i=Math.floor((x-h.origin[0])/h.resolution),j=Math.floor((y-h.origin[1])/h.resolution);return i>=0&&j>=0&&i<h.width&&j<h.height?j*h.width+i:-1;}
function walkable(x,y){const h=env?.hazard,k=gridIndex(x,y,h);return k>=0&&h.mask[k];}
function safeSegment(x,y,nx,ny){const n=Math.max(1,Math.ceil(Math.hypot(nx-x,ny-y)/.15));for(let i=1;i<=n;i++)if(!walkable(x+(nx-x)*i/n,y+(ny-y)*i/n))return false;return true;}
function buildHazard(){
 const h=env?.hazard;if(!h)return;hazardImage.width=h.width;hazardImage.height=h.height;const c=hazardImage.getContext('2d'),im=c.createImageData(h.width,h.height);
 for(let k=0;k<h.concentration.length;k++){const q=h.concentration[k];if(!h.mask[k]||q<.0001)continue;const j=Math.floor(k/h.width),i=k%h.width,o=((h.height-1-j)*h.width+i)*4;im.data[o]=255;im.data[o+1]=121;im.data[o+2]=75;im.data[o+3]=Math.min(130,Math.sqrt(q)*90);}c.putImageData(im,0,0);hazardDirty=false;
}
function buildDensity(){
 if(!env||!state)return;const res=1,nx=64,ny=48,values=new Float32Array(nx*ny);for(const a of state.agents){for(let j=Math.max(0,Math.floor(a.y-3));j<Math.min(ny,a.y+3);j++)for(let i=Math.max(0,Math.floor(a.x-3));i<Math.min(nx,a.x+3);i++){const d=(i+.5-a.x)**2+(j+.5-a.y)**2;values[j*nx+i]+=Math.exp(-d/2);}}
 densityImage.width=nx;densityImage.height=ny;const c=densityImage.getContext('2d'),im=c.createImageData(nx,ny);for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){if(!walkable(i+.5,j+.5))continue;const v=values[j*nx+i],k=((ny-1-j)*nx+i)*4;im.data[k]=v>2?255:116;im.data[k+1]=v>3?100:209;im.data[k+2]=v>2?55:147;im.data[k+3]=Math.min(160,v*35);}c.putImageData(im,0,0);densityDirty=false;
}
function drawImageWorld(c,img,x,y,w,h){c.save();c.translate(x,y+h);c.scale(1,-1);c.drawImage(img,0,0,w,h);c.restore();}
function advanceParticles(dt){
 const h=env?.hazard;if(!h||dt<=0)return;
 if(particles.length<170){for(let k=0;k<6;k++){const x=Math.random()*64,y=Math.random()*48;if(walkable(x,y))particles.push({x,y,age:Math.random()*5,life:4+Math.random()*8,kind:'wind',px:x,py:y});}}
 if(h.enabled&&h.strength>0){const count=Math.min(30,Math.ceil(dt*20)),source=h.source_positions[0];for(let k=0;k<count;k++){const x=source[0]+(Math.random()-.5)*.5,y=source[1]+(Math.random()-.5)*.5;if(walkable(x,y))particles.push({x,y,age:0,life:8+Math.random()*12,kind:'hazard',px:x,py:y});}}
 const sub=Math.max(1,Math.ceil(dt/.05)),sdt=dt/sub;
 for(const p of particles){p.px=p.x;p.py=p.y;p.age+=dt;for(let k=0;k<sub;k++){const idx=gridIndex(p.x,p.y,h);if(idx<0||!h.mask[idx]){p.age=p.life;break;}const diffusion=p.kind==='hazard'?Math.sqrt(sdt)*.55:0,nx=p.x+h.velocity_x[idx]*sdt+(Math.random()-.5)*diffusion,ny=p.y+h.velocity_y[idx]*sdt+(Math.random()-.5)*diffusion;if(safeSegment(p.x,p.y,nx,ny)){p.x=nx;p.y=ny;}else if(safeSegment(p.x,p.y,nx,p.y))p.x=nx;else if(safeSegment(p.x,p.y,p.x,ny))p.y=ny;}}
 particles=particles.filter(p=>p.age<p.life).slice(-700);
}
function drawExits(c){
 for(const e of state.exits){const color=e.open?colors[e.id]:'#f5836b';c.strokeStyle=color;c.lineWidth=.3;c.setLineDash(e.open?[]:[.35,.2]);c.beginPath();if(e.axis===0){c.moveTo(e.x,e.y-3);c.lineTo(e.x,e.y+3);}else{c.moveTo(e.x-3,e.y);c.lineTo(e.x+3,e.y);}c.stroke();c.setLineDash([]);
 const ex=e.x+(e.axis===0?e.sign*1.5:0),ey=e.y+(e.axis===1?e.sign*1.5:0);textAt(c,(e.open?'EXIT ':'CLOSED ')+e.id[0].toUpperCase(),ex,ey,9,color);}
}
function drawFly(c,a,t){
 const highlighted=a.id===selected;const color=a.exposure>1?'#ff9f76':a.state==='waiting'?'#e7bb7b':'#d2f6a3';
 c.save();c.translate(a.x,a.y);if(highlighted){c.beginPath();c.arc(0,0,.85,0,Math.PI*2);c.lineWidth=.09;c.strokeStyle='#e4ffc6';c.stroke();}
 c.rotate(a.heading||0);const flutter=Math.sin((state.time*22+a.id)*Math.min(1,a.speed+.1))*.04;
 c.fillStyle='#c4ddba80';c.beginPath();c.ellipse(-.08,.19+flutter,.25,.11,.55,0,Math.PI*2);c.ellipse(-.08,-.19-flutter,.25,.11,-.55,0,Math.PI*2);c.fill();c.fillStyle=color;c.beginPath();c.ellipse(0,0,.27,.13,0,0,Math.PI*2);c.fill();c.fillStyle='#213520';c.beginPath();c.arc(.17,0,.075,0,Math.PI*2);c.fill();
 if(zoom>1.6||highlighted){c.strokeStyle='#afcd96';c.lineWidth=.035;for(let j=-1;j<=1;j++){c.beginPath();c.moveTo(j*.13,.08);c.lineTo(j*.17+.08,.26);c.moveTo(j*.13,-.08);c.lineTo(j*.17+.08,-.26);c.stroke();}}
 if($('showVectors').checked&&a.speed>.1){c.strokeStyle='#a2d3b8aa';c.lineWidth=.055;c.beginPath();c.moveTo(.35,0);c.lineTo(.35+a.speed*.65,0);c.lineTo(.15+a.speed*.65,.12);c.stroke();}c.restore();
}
function render(now){
 if(baseDirty)drawBase();const d=canvas.width/W;ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,W,H);ctx.drawImage(base,0,0,W,H);
 if(env&&state){ctx.save();worldTransform(ctx);const h=env.hazard;
  if(hazardDirty)buildHazard();if(h?.concentration)drawImageWorld(ctx,hazardImage,h.origin[0],h.origin[1],h.width*h.resolution,h.height*h.resolution);
  if($('showDensity').checked){if(densityDirty)buildDensity();drawImageWorld(ctx,densityImage,0,0,64,48);}
  const pd=Math.max(0,state.time-lastParticleTime);advanceParticles(Math.min(.5,pd));lastParticleTime=state.time;
  if($('showWind').checked){for(const p of particles){const alpha=Math.min(1,p.age,.8*(p.life-p.age));ctx.globalAlpha=Math.max(0,alpha)*(p.kind==='hazard'?.8:.3);ctx.strokeStyle=p.kind==='hazard'?'#ffb48a':'#9bc9ad';ctx.lineWidth=p.kind==='hazard'?.07:.045;ctx.beginPath();ctx.moveTo(p.px,p.py);ctx.lineTo(p.x+.02,p.y+.01);ctx.stroke();}ctx.globalAlpha=1;}
  if(h?.strength>0){const [x,y]=h.source_positions[0];ctx.strokeStyle='#ffac7f';ctx.lineWidth=.1;ctx.beginPath();ctx.arc(x,y,.55,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#ffbe92';ctx.beginPath();ctx.arc(x,y,.13,0,Math.PI*2);ctx.fill();textAt(ctx,'REPELLENT',x,y-1.1,8,'#e9aa81');}
  if($('showTrails').checked){ctx.lineWidth=.055;for(const [id,points]of trails){ctx.strokeStyle=id===selected?'#d3f9a8':'#b5e59430';ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();}}
  drawExits(ctx);renderAgents.clear();const mix=Math.min(1,(now-receivedAt)/transitionMs);
  for(const a of state.agents){const old=previousAgents.get(a.id),v=old?{...a,x:old.x+(a.x-old.x)*mix,y:old.y+(a.y-old.y)*mix}:a;renderAgents.set(a.id,v);drawFly(ctx,v,now);}
  ctx.restore();
 }
 requestAnimationFrame(render);
}
function drawChart(){
 const r=chart.getBoundingClientRect();if(!r.width)return;const d=Math.min(2,devicePixelRatio||1);chart.width=r.width*d;chart.height=r.height*d;const c=chart.getContext('2d');c.scale(d,d);const w=r.width,h=r.height,left=27,bottom=17,top=8,p=state?.population||200,maxT=Math.max(30,chartHistory.at(-1)?.t||0),px=t=>left+(w-left-9)*t/maxT,py=n=>h-bottom-(h-bottom-top)*n/p;
 c.font='8px "Segoe UI"';c.textAlign='right';for(let k=0;k<=2;k++){const y=py(p*k/2);c.strokeStyle='#344736';c.lineWidth=.6;c.beginPath();c.moveTo(left,y);c.lineTo(w,y);c.stroke();c.fillStyle='#809a82';c.fillText(Math.round(p*k/2),left-7,y+3);}
 c.textAlign='center';for(let k=0;k<=3;k++){const t=maxT*k/3;c.fillText(Math.round(t)+'s',px(t),h-2);}
 if(chartHistory.length){c.beginPath();c.moveTo(px(0),py(0));chartHistory.forEach(v=>c.lineTo(px(v.t),py(v.n)));c.lineTo(px(chartHistory.at(-1).t),py(0));c.closePath();const g=c.createLinearGradient(0,top,0,h);g.addColorStop(0,'#b5ed7e40');g.addColorStop(1,'#b5ed7e00');c.fillStyle=g;c.fill();c.beginPath();c.moveTo(px(0),py(0));chartHistory.forEach(v=>c.lineTo(px(v.t),py(v.n)));c.strokeStyle='#c4ef8e';c.lineWidth=1.8;c.stroke();const v=chartHistory.at(-1);c.fillStyle='#d4f8a6';c.beginPath();c.arc(px(v.t),py(v.n),2.8,0,Math.PI*2);c.fill();}
}
async function reset(preset=activePreset){
 const config={population:Number($('population').value),seed:Number($('seed').value),preset:preset==='plume'?'open':preset,wind_direction_deg:Number($('windDirection').value),wind_speed_mps:Number($('windSpeed').value),hazard_strength:preset==='plume'?1.5:0};
 const s=await api('reset',config);activePreset=preset;trails.clear();particles=[];chartHistory=[];previousAgents.clear();renderAgents.clear();lastChartSecond=-1;lastParticleTime=0;selected=null;requestSeconds=0;requestWall=0;simBudget=0;accept(s,true);await environment();document.querySelectorAll('.scenario').forEach(b=>b.classList.toggle('active',b.dataset.preset===preset));running=true;setPlay();
}
$('play').onclick=()=>{running=!running;simBudget=0;setPlay();};$('reset').onclick=()=>action(()=>reset());$('apply').onclick=()=>action(()=>reset());
$('alarm').onclick=()=>action(async()=>{accept(await api('control',{alarm:true}));running=true;setPlay();});
$('population').oninput=()=>$('populationValue').textContent=$('population').value+' flies';
document.querySelectorAll('.scenario').forEach(b=>b.onclick=()=>action(()=>reset(b.dataset.preset)));
for(const id of ['windDirection','windSpeed']){
 $(id).oninput=updateWindLabels;
 $(id).onchange=()=>{
  const values={wind_direction_deg:Number($('windDirection').value),wind_speed_mps:Number($('windSpeed').value)};
  action(async()=>{accept(await api('control',values));await environment();});
 };
}
$('hazardStrength').oninput=updateWindLabels;$('hazardStrength').onchange=()=>{const strength=Number($('hazardStrength').value);action(async()=>{accept(await api('control',{hazard_strength:strength}));await environment();});};
$('placeHazard').onclick=()=>{placement=!placement;$('placementHint').hidden=!placement;$('placeHazard').classList.toggle('active',placement);};
$('closeSelection').onclick=()=>{selected=null;updateSelection();};
canvas.onclick=e=>{if(!env||!state)return;const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left-transform.x)/transform.s,y=(transform.y-e.clientY+r.top)/transform.s;if(placement){if(!walkable(x,y)){$('placementHint').textContent='Choose an open floor or passage';return;}action(async()=>{accept(await api('control',{hazard_position:[x,y]}));if(!state.config.hazard_strength){accept(await api('control',{hazard_strength:1.5}),true);}await environment();placement=false;$('placementHint').hidden=true;$('placementHint').textContent='Click the arena to place the repellent source';$('placeHazard').classList.remove('active');});return;}let best=null,d=1.2;for(const a of state.agents){const dist=Math.hypot(a.x-x,a.y-y);if(dist<d){best=a.id;d=dist;}}selected=best;updateSelection();};
function changeZoom(f){zoom=Math.max(.75,Math.min(4,zoom*f));baseDirty=true;}
$('zoomIn').onclick=()=>changeZoom(1.25);$('zoomOut').onclick=()=>changeZoom(.8);$('fit').onclick=()=>{zoom=1;panX=panY=0;baseDirty=true;};
canvas.addEventListener('wheel',e=>{e.preventDefault();changeZoom(e.deltaY<0?1.1:1/1.1);},{passive:false});
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.arena-panel').requestFullscreen();}catch(e){error(e);}};
$('detailsButton').onclick=()=>$('modelDialog').showModal();$('closeDetails').onclick=()=>$('modelDialog').close();
$('export').onclick=()=>action(async()=>{const data=await api('export');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='arena-experiment-'+data.state.config.seed+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
async function tick(){
 const now=performance.now(),elapsedWall=(now-lastTick)/1000,delta=Math.min(.15,elapsedWall);lastTick=now;
 if(running&&state&&state.phase!=='complete'){simBudget=Math.min(.5,simBudget+delta*Number($('speed').value));requestWall+=elapsedWall;}else simBudget=0;
 if(!pending&&simBudget>=.05){const steps=Math.min(10,Math.floor(simBudget/.05));simBudget-=steps*.05;pending=true;action(async()=>{try{const next=await api('step',{steps,dt:.05});requestSeconds+=next.time-state.time;accept(next);if(state.time-lastEnvironment>=.8)await environment();}finally{pending=false;}});}
 setTimeout(tick,25);
}
action(async()=>{accept(await api('state'),true);await environment();resize();clearError();lastParticleTime=state.time;requestAnimationFrame(render);tick();});

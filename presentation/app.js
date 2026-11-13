const colors = {
  ink: "#f5f3eb",
  muted: "#aaa9a2",
  line: "rgba(245,243,235,.14)",
  green: "#a8f07a",
  coral: "#ff806d",
  blue: "#7cc8ff",
  purple: "#b6a0ff",
  gold: "#f4c96b",
};

const matchedData = [
  {scene:"Drums", budget:.20, mask:"none", penalty:0, psnr_db:21.735374, ssim:.890939, lpips:.092615},
  {scene:"Drums", budget:.20, mask:".002", penalty:.002, psnr_db:20.790253, ssim:.870971, lpips:.115026},
  {scene:"Drums", budget:.20, mask:".004", penalty:.004, psnr_db:20.826793, ssim:.868513, lpips:.115856},
  {scene:"Drums", budget:.20, mask:".008", penalty:.008, psnr_db:21.364698, ssim:.873926, lpips:.109170},
  {scene:"Drums", budget:.40, mask:"none", penalty:0, psnr_db:22.127879, ssim:.896491, lpips:.084321},
  {scene:"Drums", budget:.40, mask:".002", penalty:.002, psnr_db:21.623359, ssim:.887968, lpips:.093752},
  {scene:"Drums", budget:.40, mask:".004", penalty:.004, psnr_db:21.536264, ssim:.883414, lpips:.101613},
  {scene:"Drums", budget:.40, mask:".008", penalty:.008, psnr_db:21.635929, ssim:.879858, lpips:.102663},
  {scene:"Ficus", budget:.20, mask:"none", penalty:0, psnr_db:21.274762, ssim:.901266, lpips:.077812},
  {scene:"Ficus", budget:.20, mask:".002", penalty:.002, psnr_db:21.296157, ssim:.899633, lpips:.077515},
  {scene:"Ficus", budget:.20, mask:".004", penalty:.004, psnr_db:22.311245, ssim:.908392, lpips:.066911},
  {scene:"Ficus", budget:.40, mask:"none", penalty:0, psnr_db:21.567623, ssim:.905704, lpips:.072812},
  {scene:"Ficus", budget:.40, mask:".002", penalty:.002, psnr_db:21.519471, ssim:.903457, lpips:.073639},
  {scene:"Ficus", budget:.40, mask:".004", penalty:.004, psnr_db:22.446403, ssim:.910975, lpips:.065918},
  {scene:"Lego", budget:.20, mask:"none", penalty:0, psnr_db:26.168691, ssim:.936263, lpips:.048526},
  {scene:"Lego", budget:.20, mask:".002", penalty:.002, psnr_db:24.045960, ssim:.912138, lpips:.073571},
  {scene:"Lego", budget:.20, mask:".004", penalty:.004, psnr_db:23.820210, ssim:.906032, lpips:.081555},
  {scene:"Lego", budget:.20, mask:".008", penalty:.008, psnr_db:23.129542, ssim:.885971, lpips:.106054},
  {scene:"Lego", budget:.40, mask:"none", penalty:0, psnr_db:26.799071, ssim:.942566, lpips:.040320},
  {scene:"Lego", budget:.40, mask:".002", penalty:.002, psnr_db:25.330856, ssim:.928021, lpips:.054684},
  {scene:"Lego", budget:.40, mask:".004", penalty:.004, psnr_db:24.690319, ssim:.917317, lpips:.067680},
  {scene:"Lego", budget:.40, mask:".008", penalty:.008, psnr_db:24.007994, ssim:.899984, lpips:.091299},
];

const maskColor = {none: colors.ink, ".002": colors.blue, ".004": colors.green, ".008": colors.purple};
const metricMeta = {
  psnr_db: {label:"PSNR", unit:"dB", better:"higher", digits:2},
  ssim: {label:"SSIM", unit:"", better:"higher", digits:3},
  lpips: {label:"LPIPS", unit:"", better:"lower", digits:3},
};

let matchedState = {budget:.40, metric:"psnr_db"};

function svgText(x, y, value, className="", anchor="start") {
  return `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;
}

function renderMatchedChart() {
  const root = document.getElementById("matched-chart");
  const metric = matchedState.metric;
  const rows = matchedData.filter(d => d.budget === matchedState.budget);
  const values = rows.map(d => d[metric]);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const pad = Math.max((rawMax - rawMin) * .12, metric === "psnr_db" ? .25 : .002);
  const min = rawMin - pad;
  const max = rawMax + pad;
  const W = 1120, H = 410, left = 125, right = 75, top = 50, bottom = 42;
  const chartW = W-left-right;
  const x = v => left + ((v-min)/(max-min))*chartW;
  const scenes = ["Drums","Ficus","Lego"];
  const groupH = 104;
  let marks = "";
  let labels = "";
  const ticks = 5;
  for (let i=0;i<=ticks;i++) {
    const v = min + (max-min)*(i/ticks);
    const px=x(v);
    marks += `<line x1="${px}" y1="${top-16}" x2="${px}" y2="${H-bottom}" class="grid-line"/>`;
    labels += svgText(px,H-12,v.toFixed(metricMeta[metric].digits),"", "middle");
  }
  scenes.forEach((scene,si) => {
    const sceneRows=rows.filter(d=>d.scene===scene);
    const cy=top+si*groupH+30;
    labels += svgText(0,cy+7,scene,"axis-label");
    marks += `<line x1="${left}" y1="${cy}" x2="${W-right}" y2="${cy}" stroke="rgba(245,243,235,.08)"/>`;
    sceneRows.forEach((d,mi)=>{
      const py=cy-18+mi*12;
      const px=x(d[metric]);
      const winner = metric === "lpips" ? d[metric] === Math.min(...sceneRows.map(r=>r[metric])) : d[metric] === Math.max(...sceneRows.map(r=>r[metric]));
      marks += `<circle cx="${px}" cy="${py}" r="${winner?7:5}" fill="${maskColor[d.mask]}" ${winner?'stroke="#090d0d" stroke-width="2"':''}/>`;
      labels += svgText(px+10,py+4,`${d.mask==='none'?'importance only':'λ '+d.mask} · ${d[metric].toFixed(metricMeta[metric].digits)}`,winner?"value-label":"");
    });
  });
  const title=`${metricMeta[metric].label} at ${Math.round(matchedState.budget*100)}% retention; ${metricMeta[metric].better} is better.`;
  root.innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-labelledby="matched-svg-title matched-svg-desc"><title id="matched-svg-title">${title}</title><desc id="matched-svg-desc">Dot plot comparing importance-only and learned-mask settings for Drums, Ficus, and Lego.</desc>${marks}${labels}${svgText(W-right,26,`${metricMeta[metric].label} ${metricMeta[metric].unit}`,"axis-label","end")}</svg>`;
  root.setAttribute("aria-label", title);
  const budgetLabel=Math.round(matchedState.budget*100);
  document.getElementById("matched-insight").textContent = `At ${budgetLabel}% retention: Ficus λ .004 wins; Drums and Lego favour importance only.`;
}

function renderDeltaChart() {
  const root=document.getElementById("delta-chart");
  const data=[
    {scene:"Drums", b20:-.371, b40:-.505},
    {scene:"Ficus", b20:1.036, b40:.879},
    {scene:"Lego", b20:-2.123, b40:-1.468},
  ];
  const W=1120,H=210,left=130,right=65,top=30,rowH=58;
  const min=-2.4,max=1.25,zero=left+((-min)/(max-min))*(W-left-right);
  const x=v=>left+((v-min)/(max-min))*(W-left-right);
  let out=`<line x1="${zero}" y1="10" x2="${zero}" y2="${H-22}" class="zero-line"/>`;
  data.forEach((d,i)=>{
    const y=top+i*rowH;
    out+=svgText(0,y+17,d.scene,"axis-label");
    [[d.b40,"40%",y],[d.b20,"20%",y+22]].forEach(([v,label,py])=>{
      const end=x(v), bx=Math.min(zero,end), bw=Math.abs(end-zero);
      out+=`<rect x="${bx}" y="${py}" width="${Math.max(2,bw)}" height="9" rx="4.5" fill="${v>=0?colors.green:colors.coral}"/>`;
      out+=svgText(v>=0?end+8:end-8,py+8,`${label}  ${v>0?'+':''}${v.toFixed(2)} dB`,"value-label",v>=0?"start":"end");
    });
  });
  out+=svgText(zero,H-2,"importance-only baseline","","middle");
  root.innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-labelledby="delta-title delta-desc"><title id="delta-title">PSNR change versus importance-only pruning</title><desc id="delta-desc">Ficus improves at both budgets; Drums and Lego decline.</desc>${out}</svg>`;
}

function renderCollapseChart() {
  const root=document.getElementById("collapse-chart");
  const penalties=[0,.002,.004,.008,.016];
  const series={Drums:[300000,300000,300000,300000,16],Ficus:[300000,300000,300000,39,0],Lego:[300000,300000,300000,300000,84]};
  const seriesColor={Drums:colors.blue,Ficus:colors.green,Lego:colors.gold};
  const W=1120,H=390,left=110,right=90,top=40,bottom=60;
  const x=i=>left+i*(W-left-right)/(penalties.length-1);
  const y=v=>top+(300000-v)/300000*(H-top-bottom);
  let out="";
  [0,60000,120000,180000,240000,300000].forEach(v=>{
    const py=y(v);
    out+=`<line x1="${left}" y1="${py}" x2="${W-right}" y2="${py}" class="${v===60000||v===120000?'target-line':'grid-line'}"/>`;
    out+=svgText(left-15,py+5,v===0?"0":`${v/1000}k`,"", "end");
    if(v===60000||v===120000) out+=svgText(W-right+10,py+4,`${v/300000*100}% target`,"","start");
  });
  penalties.forEach((p,i)=>{out+=svgText(x(i),H-22,p===0?"none":p.toFixed(3),"","middle");});
  Object.entries(series).forEach(([name,values])=>{
    const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(" ");
    out+=`<polyline points="${points}" fill="none" stroke="${seriesColor[name]}" stroke-width="4" stroke-linejoin="round"/>`;
    values.forEach((v,i)=>{out+=`<circle cx="${x(i)}" cy="${y(v)}" r="6" fill="${seriesColor[name]}" stroke="#090d0d" stroke-width="2"/>`;});
    const final=values.at(-1);
    out+=svgText(x(values.length-1)+15,y(final)+(name==="Ficus"?18:name==="Drums"?-18:2),`${name} · ${final}`,"value-label");
  });
  out+=svgText(W/2,H,"mask penalty λ","axis-label","middle");
  root.innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-labelledby="collapse-title collapse-desc"><title id="collapse-title">Final Gaussian count by mask penalty</title><desc id="collapse-desc">All scenes remain at capacity under mild penalties, then abruptly collapse; Ficus collapses one step earlier.</desc>${out}</svg>`;
}

function renderRobustnessChart() {
  const root=document.getElementById("robustness-chart");
  const data=[
    {scene:"Drums",before:22.154541,after:21.950661,color:colors.blue},
    {scene:"Ficus",before:21.599721,after:21.395604,color:colors.green},
    {scene:"Lego",before:26.777542,after:26.484508,color:colors.gold},
  ];
  const W=700,H=470,left=135,right=110,top=60,bottom=55;
  const min=20.8,max=27.2;
  const y=v=>top+(max-v)/(max-min)*(H-top-bottom);
  const x1=left+80,x2=W-right;
  let out=`<line x1="${x1}" y1="${top-20}" x2="${x1}" y2="${H-bottom+10}" class="grid-line"/><line x1="${x2}" y1="${top-20}" x2="${x2}" y2="${H-bottom+10}" class="grid-line"/>`;
  out+=svgText(x1,24,"300k Gaussians","axis-label","middle")+svgText(x2,24,"75k Gaussians","axis-label","middle");
  data.forEach(d=>{
    const y1=y(d.before),y2=y(d.after),drop=d.before-d.after;
    out+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${d.color}" stroke-width="4"/>`;
    out+=`<circle cx="${x1}" cy="${y1}" r="7" fill="${d.color}"/><circle cx="${x2}" cy="${y2}" r="7" fill="${d.color}"/>`;
    out+=svgText(x1-14,y1+5,`${d.scene}  ${d.before.toFixed(2)}`,"value-label","end");
    out+=svgText(x2+14,y2+5,`${d.after.toFixed(2)}  (−${drop.toFixed(2)})`,"value-label","start");
  });
  root.innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-labelledby="robust-title robust-desc"><title id="robust-title">Importance-only PSNR before and after pruning 75 percent</title><desc id="robust-desc">All three scenes lose less than 0.3 decibels while retaining one quarter of the Gaussians.</desc>${out}</svg>`;
}

function setupControls() {
  document.querySelectorAll("[data-budget]").forEach(button=>button.addEventListener("click",()=>{
    matchedState.budget=Number(button.dataset.budget);
    document.querySelectorAll("[data-budget]").forEach(b=>{const active=b===button;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});
    renderMatchedChart();
  }));
  document.querySelectorAll("[data-metric]").forEach(button=>button.addEventListener("click",()=>{
    matchedState.metric=button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach(b=>{const active=b===button;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});
    renderMatchedChart();
  }));
}

function setupPresentationNavigation() {
  const slides=[...document.querySelectorAll(".slide")];
  const links=[...document.querySelectorAll(".slide-nav a")];
  const counter=document.getElementById("slide-counter");
  const progress=document.getElementById("progress-bar");
  let current=0;
  const setCurrent=i=>{
    current=i;
    counter.textContent=`${String(i+1).padStart(2,"0")} / ${String(slides.length).padStart(2,"0")}`;
    progress.style.width=`${((i+1)/slides.length)*100}%`;
    links.forEach((link,index)=>link.classList.toggle("active",index===i));
  };
  const observer=new IntersectionObserver(entries=>{
    entries.forEach(entry=>{if(entry.isIntersecting) setCurrent(slides.indexOf(entry.target));});
  },{threshold:.58});
  slides.forEach(slide=>observer.observe(slide));
  document.addEventListener("keydown",event=>{
    if(["INPUT","BUTTON","SELECT"].includes(document.activeElement.tagName)) return;
    if(["ArrowDown","ArrowRight","PageDown"," "].includes(event.key)){event.preventDefault();slides[Math.min(current+1,slides.length-1)].scrollIntoView();}
    if(["ArrowUp","ArrowLeft","PageUp"].includes(event.key)){event.preventDefault();slides[Math.max(current-1,0)].scrollIntoView();}
    if(event.key==="Home"){event.preventDefault();slides[0].scrollIntoView();}
    if(event.key==="End"){event.preventDefault();slides.at(-1).scrollIntoView();}
  });
  document.getElementById("fullscreen-button").addEventListener("click",async()=>{
    if(!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  });
  setCurrent(0);
}

renderMatchedChart();
renderDeltaChart();
renderCollapseChart();
renderRobustnessChart();
setupControls();
setupPresentationNavigation();

/* ============ Small helpers ============ */
const fmt=(x,k=6)=>Number.isFinite(x)?Number(x).toFixed(k):String(x);
const zeros=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));
const eye=n=>{const I=zeros(n,n);for(let i=0;i<n;i++)I[i][i]=1;return I;};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const matVec=(A,x)=>A.map(r=>dot(r,x));
const vecSub=(a,b)=>a.map((v,i)=>v-b[i]);
const norm2=v=>Math.sqrt(dot(v,v));
const copy=A=>A.map(r=>r.slice());
const els={};
const state={workspace:'root',methodId:null};
class Recommendation extends Error{}
function fixedWidthMatrix(M,k=6){
  const s=M.map(r=>r.map(v=>fmt(v,k)));
  const w=Math.max(...s.flat().map(t=>t.length));
  return s.map(r=>' '+r.map(t=>t.padStart(w,' ')).join(' ')+' ').join('\\n');
}
function statusBadge(kind,msg){
  if(!els.status) return;
  const cls=kind==='ok'?'badge-ok':kind==='warn'?'badge-warn':'badge-err';
  els.status.innerHTML = '<div class="card"><span class="badge '+cls+'">'+kind.toUpperCase()+
    '</span> <span class="ml-2">'+msg+'</span></div>';
}

/* ============ Normalization & Parsers ============ */
// Normalize typical Unicode math symbols to ASCII and report first offending char
function normalizeExpr(src){
  if(typeof src!=='string') return '';
  src = src.normalize('NFKC');
  // dashes/minus to '-'
  src = src.replace(/[\\u2212\\u2012-\\u2015]/g,'-');
  // times / middle dots to '*'
  src = src.replace(/[\\u00D7\\u2715\\u2716\\u22C5\\u00B7]/g,'*');
  // fancy division slash to '/'
  src = src.replace(/[\\u2215]/g,'/');
  return src;
}
function firstInvalidChar(str){
  const allowed=/[0-9a-zA-Z_\\s+\\-*/^().,]/;
  for(const ch of str){
    if(!allowed.test(ch)){
      const cp = ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');
      return {ch,cp};
    }
  }
  return null;
}
function parseFun(id){
  let raw=document.getElementById(id).value||'';
  raw = normalizeExpr(raw).trim();
  // allow ^ as power (convert to **)
  raw = raw.replaceAll('^','**');
  // Map 'pi' a Math.PI via with(Math){...}
  raw = raw.replace(/\\bpi\\b/ig,'PI');
  if(!/[xX]/.test(raw)) throw Error('Function must contain x');
  const bad = firstInvalidChar(raw);
  if(bad){
    throw Error(`Invalid character: «${bad.ch}» (U+${bad.cp}). Allowed: digits, x, + - * / ^ ( ) . , and names like sin, cos, exp, log, sqrt, PI.`);
  }
  const f=new Function('x','with(Math){return '+raw+'}');
  try{ void f(0); }catch(e){ throw Error('Function is not valid: '+e.message); }
  return {src:raw,f};
}
function parseNum(id,label=id){
  const el=document.getElementById(id); if(!el) throw Error('Missing input: '+label);
  const raw=el.value.trim().replace(/,/g,'');
  if(!/^[-]?(\\d+(\\.\\d*)?|\\.\\d+)(e[-+]?\\d+)?$/i.test(raw))
    throw Error(`Invalid number in ${label}: “${el.value}”`);
  return Number(raw);
}
function parseVec(id,label=id){
  const raw=document.getElementById(id).value.trim(); if(!raw) throw Error(label+' is empty');
  const parts=raw.split(/[ ,\\t\\n]+/).filter(Boolean);
  const arr=parts.map(Number);
  const badIndex=arr.findIndex(v=>!Number.isFinite(v));
  if(badIndex!==-1) throw Error(`${label} has an invalid entry at position ${badIndex+1}: “${parts[badIndex]}”`);
  return arr;
}
function parseMat(id,label=id){
  const raw=document.getElementById(id).value.trim(); if(!raw) throw Error(label+' is empty');
  const rows=raw.split(/\\n|;/).map(r=>r.trim()).filter(Boolean);
  const M=rows.map(r=>r.split(/[ ,\\t]+/).filter(Boolean).map(Number));
  const m=M[0].length;
  if(M.some(r=>r.length!==m)) throw Error(label+' must be rectangular (same number of columns per row)');
  const flat=M.flat(), badIndex=flat.findIndex(v=>!Number.isFinite(v));
  if(badIndex!==-1) throw Error(`${label} has a non-numeric value near row ${Math.floor(badIndex/m)+1}`);
  return M;
}
function parsePairs(id,label=id){
  const raw=document.getElementById(id).value.trim(); if(!raw) throw Error(label+' is empty');
  const lines=raw.split(/\n+/).map(r=>r.trim()).filter(Boolean);
  const X=[],Y=[];
  lines.forEach((line,idx)=>{
    const parts=line.split(/[ ,;\t]+/).filter(Boolean);
    if(parts.length<2) throw Error(`${label} line ${idx+1} must contain x;y`);
    const x=Number(parts[0]), y=Number(parts[1]);
    if(!Number.isFinite(x)||!Number.isFinite(y)) throw Error(`${label} line ${idx+1} has non-numeric data`);
    X.push(x); Y.push(y);
  });
  return {X,Y};
}

/* ============ Linear algebra helpers ============ */
function fwdSub(L,b){const n=L.length,y=Array(n).fill(0);for(let i=0;i<n;i++){let s=0;for(let j=0;j<i;j++)s+=L[i][j]*y[j];if(Math.abs(L[i][i])<1e-14)throw Error('Zero on diagonal in forward substitution');y[i]=(b[i]-s)/L[i][i];}return y;}
function bwdSub(U,y){const n=U.length,x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=0;for(let j=i+1;j<n;j++)s+=U[i][j]*x[j];if(Math.abs(U[i][i])<1e-14)throw Error('Zero on diagonal in backward substitution');x[i]=(y[i]-s)/U[i][i];}return x;}
function matMul(A,B){const n=A.length,m=B[0].length,k=B.length,C=zeros(n,m);for(let i=0;i<n;i++)for(let j=0;j<m;j++){let s=0;for(let t=0;t<k;t++)s+=A[i][t]*B[t][j];C[i][j]=s;}return C;}
function inv(M){const n=M.length,A=copy(M),I=eye(n);for(let i=0;i<n;i++){let piv=i,mv=Math.abs(A[i][i]);for(let r=i+1;r<n;r++)if(Math.abs(A[r][i])>mv){mv=Math.abs(A[r][i]);piv=r;}if(piv!==i){[A[i],A[piv]]=[A[piv],A[i]];[I[i],I[piv]]=[I[piv],I[i]];}const d=A[i][i];if(Math.abs(d)<1e-14)throw Error('Singular matrix');for(let j=0;j<n;j++){A[i][j]/=d;I[i][j]/=d;}for(let r=0;r<n;r++)if(r!==i){const m=A[r][i];for(let j=0;j<n;j++){A[r][j]-=m*A[i][j];I[r][j]-=m*I[i][j];}}}return I;}
function spectralRadius(T,max=200){const n=T.length;let v=Array(n).fill(0).map((_,i)=>i===0?1:0),l=0;for(let k=0;k<max;k++){const w=matVec(T,v),nrm=norm2(w);if(nrm===0)return 0;v=w.map(z=>z/nrm);const Tv=matVec(T,v);l=dot(v,Tv);}return Math.abs(l);}

/* ============ Plots ============ */
const Plots={
  blank(div,msg){Plotly.newPlot(div,[{x:[0],y:[0],mode:'markers',marker:{opacity:0}}],{annotations:[{text:msg,showarrow:false}],margin:{t:20},showlegend:false});},
  func(div,f,range,marks){const [L,R]=[Math.min(...range),Math.max(...range)];const xs=[],ys=[];const n=240;for(let i=0;i<=n;i++){const x=L+(R-L)*i/n;xs.push(x);ys.push(f(x));}const data=[{x:xs,y:ys,mode:'lines',type:'scatter',name:'f(x)'}];if(marks&&marks.length){data.push({x:marks,y:marks.map(_=>0),mode:'markers+text',text:marks.map(x=>fmt(x,4)),textposition:'top center',name:'iter'});}Plotly.newPlot(div,data,{margin:{t:20},showlegend:false});},
  series(div,arr,title){Plotly.newPlot(div,[{x:arr.map((_,i)=>i+1),y:arr,mode:'lines+markers+text',text:arr.map(v=>fmt(v,3)),textposition:'top center'}],{margin:{t:20},xaxis:{title:'k'},yaxis:{title:'||e||'},title});},
  scatterFit(div,pts,poly,spl){const {X,Y}=pts;const data=[{x:X,y:Y,mode:'markers+text',text:Y.map(v=>fmt(v,3)),textposition:'top center',type:'scatter',name:'data'}];if(poly){const n=poly.length;const xs=[...Array(240).keys()].map(i=>X[0]+(X.at(-1)-X[0])*i/239);const ys=xs.map(x=>poly.reduce((s,c,j)=>s+c*Math.pow(x,n-1-j),0));data.push({x:xs,y:ys,mode:'lines',name:'fit'});}if(spl){for(let i=0;i<spl.length;i++){const a=X[i],b=X[i+1];const xs=[...Array(60).keys()].map(k=>a+(b-a)*k/59);let ys;if(spl[i].length===2){const[p,q]=spl[i];ys=xs.map(x=>p*x+q);}else if(spl[i].length===3){const[A,B,C]=spl[i];ys=xs.map(x=>A*x*x+B*x+C);}else{const[A,B,C,D]=spl[i];ys=xs.map(x=>((A*x+B)*x+C)*x+D);}data.push({x:xs,y:ys,mode:'lines',line:{dash:'dot'},showlegend:false});}}Plotly.newPlot(div,data,{margin:{t:20}});}
};
function mapRegistry(dict){
  return Object.entries(dict).map(([name,meta])=>({name,...meta}));
}

/* ============ Root finding (order-agnostic) ============ */
function incSearch(f,a,delta,nmax){
  if(delta===0) return {bracket:null,msg:'Δ must be non-zero.'};
  const L=Math.min(a,a-delta*nmax), R=Math.max(a,a+delta*nmax);
  const step = Math.sign(delta)||1, h=Math.abs(delta);
  let x0=a, y0=f(x0);
  for(let k=0;k<nmax;k++){
    const x1 = x0 + step*h, y1=f(x1);
    if(y0*y1<0) return {bracket:[Math.min(x0,x1),Math.max(x0,x1)], msg:`Sign change in [${fmt(Math.min(x0,x1))}, ${fmt(Math.max(x0,x1))}]`};
    x0=x1; y0=y1; if(x0<L||x0>R) break;
  }
  return {bracket:null,msg:'No sign change found on the scanned range.'};
}
function bisection(f,a,b,tol,maxIt){
  let L=Math.min(a,b), R=Math.max(a,b);
  if(f(L)*f(R)>=0) throw new Recommendation('No sign change on [a,b]. Try finding a bracket with a different interval or use Incremental Search.');
  const iters=[]; let mid,fm,FL=f(L),FR=f(R);
  for(let k=1;k<=maxIt;k++){
    mid=(L+R)/2; fm=f(mid);
    iters.push([k,L,R,mid,FL,FR,fm,Math.abs(R-L)/2]);
    if(Math.abs(fm)<=tol || Math.abs(R-L)<=2*tol) break;
    if(FL*fm<0){ R=mid; FR=fm; } else { L=mid; FL=fm; }
  }
  return {root:mid,iters,range:[L,R]};
}
function falsePosition(f,a,b,tol,maxIt){
  let L=Math.min(a,b), R=Math.max(a,b);
  if(f(L)*f(R)>=0) throw new Recommendation('No sign change on [a,b].');
  const iters=[]; let x=L, FL=f(L), FR=f(R);
  for(let k=1;k<=maxIt;k++){
    const denom=FR-FL; if(Math.abs(denom)<1e-14) throw new Recommendation('Denominator near zero; choose another interval.');
    x = R - FR*(R-L)/denom; const fx=f(x);
    iters.push([k,L,R,x,FL,FR,fx,Math.abs(R-L)]);
    if(Math.abs(fx)<=tol) break;
    if(FL*fx<0){ R=x; FR=fx; } else { L=x; FL=fx; }
  }
  return {root:x,iters,range:[L,R]};
}
function fixedPoint(g,[L,R],x0,tol,maxIt){
  const iters=[]; for(let k=1;k<=maxIt;k++){ const x1=g(x0); const err=Math.abs(x1-x0); iters.push([k,x0,x1,err]); if(err<=tol) return {root:x1,iters}; if(!(x1>=L && x1<=R)) throw new Recommendation('Iterate inside the interval; tighten [a,b] or change g(x).'); x0=x1; } return {root:x0,iters};
}
function newtonRoot(f,x0,tol,maxIt){
  const iters=[], der=x=>(f(x+1e-6)-f(x-1e-6))/(2e-6); let x=x0, fx=f(x);
  for(let k=1;k<=maxIt;k++){ const d=der(x); if(Math.abs(d)<1e-14) throw new Recommendation('Derivative near zero; try another x0.'); const x1=x - fx/d; iters.push([k,x,fx,d,x1,Math.abs(x1-x)]); if(Math.abs(x1-x)<=tol){ x=x1; fx=f(x); break; } x=x1; fx=f(x);} return {root:x,iters};
}
function secant(f,[L,R],x0,x1,tol,maxIt){
  const iters=[]; for(let k=1;k<=maxIt;k++){ const f0=f(x0), f1=f(x1), denom=f1-f0; if(Math.abs(denom)<1e-14) throw new Recommendation('Denominator near zero; pick better seeds.'); const x2=x1 - f1*(x1-x0)/denom; if(!(x2>=L && x2<=R)) throw new Recommendation('Iterate inside [a,b]; tighten bounds or change seeds.'); const err=Math.abs(x2-x1); iters.push([k,x0,x1,x2,f(x2),err]); if(err<=tol) return {root:x2,iters}; x0=x1; x1=x2; } return {root:x1,iters};
}

/* ============ Factorizations ============ */
function fwdSub(L,b){const n=L.length,y=Array(n).fill(0);for(let i=0;i<n;i++){let s=0;for(let j=0;j<i;j++)s+=L[i][j]*y[j];if(Math.abs(L[i][i])<1e-14)throw Error('Zero on diagonal in forward substitution');y[i]=(b[i]-s)/L[i][i];}return y;}
function bwdSub(U,y){const n=U.length,x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=0;for(let j=i+1;j<n;j++)s+=U[i][j]*x[j];if(Math.abs(U[i][i])<1e-14)throw Error('Zero on diagonal in backward substitution');x[i]=(y[i]-s)/U[i][i];}return x;}
function luSimple(A){const n=A.length,L=eye(n),U=zeros(n,n),M=copy(A);for(let i=0;i<n-1;i++){for(let j=i+1;j<n;j++){if(M[j][i]!==0){const m=M[j][i]/M[i][i];L[j][i]=m;for(let k=i;k<n;k++)M[j][k]-=m*M[i][k];}}for(let k=i;k<n;k++)U[i][k]=M[i][k];}U[n-1][n-1]=M[n-1][n-1];return{L,U};}
function luPartialPivot(A){const n=A.length,L=eye(n),U=zeros(n,n),P=eye(n),M=copy(A);for(let i=0;i<n-1;i++){let piv=i,mv=Math.abs(M[i][i]);for(let r=i+1;r<n;r++)if(Math.abs(M[r][i])>mv){mv=Math.abs(M[r][i]);piv=r;}if(piv!==i){[M[i],M[piv]]=[M[piv],M[i]];[P[i],P[piv]]=[P[piv],P[i]];for(let k=0;k<i;k++)[L[i][k],L[piv][k]]=[L[piv][k],L[i][k]];}for(let j=i+1;j<n;j++)if(M[j][i]!==0){const m=M[j][i]/M[i][i];L[j][i]=m;for(let k=i;k<n;k++)M[j][k]-=m*M[i][k];}for(let k=i;k<n;k++)U[i][k]=M[i][k];}U[n-1][n-1]=M[n-1][n-1];return{L,U,P};}
function crout(A){const n=A.length,L=zeros(n,n),U=eye(n);for(let i=0;i<n;i++){for(let j=i;j<n;j++){let s=0;for(let k=0;k<i;k++)s+=L[j][k]*U[k][i];L[j][i]=A[j][i]-s;}for(let j=i+1;j<n;j++){let s=0;for(let k=0;k<i;k++)s+=L[i][k]*U[k][j];if(Math.abs(L[i][i])<1e-14)throw Error('Zero on diagonal in Crout');U[i][j]=(A[i][j]-s)/L[i][i];}}return{L,U};}
function doolittle(A){const n=A.length,L=eye(n),U=zeros(n,n);for(let i=0;i<n;i++){for(let j=i;j<n;j++){let s=0;for(let k=0;k<i;k++)s+=L[i][k]*U[k][j];U[i][j]=A[i][j]-s;}for(let j=i+1;j<n;j++){let s=0;for(let k=0;k<i;k++)s+=L[j][k]*U[k][i];if(Math.abs(U[i][i])<1e-14)throw Error('Zero on diagonal in Doolittle');L[j][i]=(A[j][i]-s)/U[i][i];}}return{L,U};}
function choleskyCourse(A){const n=A.length,L=zeros(n,n),U=zeros(n,n);if(Math.abs(A[0][0])<1e-14)throw Error('a_11 = 0; cannot start Cholesky.');for(let i=0;i<n;i++){let s=0;for(let k=0;k<i;k++)s+=L[i][k]*U[k][i];const diag=A[i][i]-s;if(diag<=0)throw new Recommendation('Non-positive pivot in Cholesky (matrix not SPD).');L[i][i]=Math.sqrt(diag);U[i][i]=L[i][i];for(let j=i+1;j<n;j++){let s2=0;for(let k=0;k<i;k++)s2+=L[j][k]*U[k][i];L[j][i]=(A[j][i]-s2)/U[i][i];}for(let j=i+1;j<n;j++){let s3=0;for(let k=0;k<i;k++)s3+=L[i][k]*U[k][j];U[i][j]=(A[i][j]-s3)/L[i][i];}}return{L,U};}

/* ============ Iteratives ============ */
function jacobi(A,b,x0,tol,Nmax){const n=A.length;for(let i=0;i<n;i++)if(Math.abs(A[i][i])<1e-14)throw Error('Jacobi: zero on diagonal');const D=eye(n).map((r,i)=>r.map((_,j)=>i===j?A[i][i]:0));const L=zeros(n,n),U=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){if(i>j)L[i][j]=-A[i][j];else if(i<j)U[i][j]=-A[i][j];}const Di=eye(n).map((r,i)=>r.map((_,j)=>i===j?1/D[i][i]:0));const S=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)S[i][j]=L[i][j]+U[i][j];const T=matMul(Di,S),C=matVec(Di,b),rho=spectralRadius(T);const iters=[];let x=x0.slice(),E=Infinity,k=0;while(E>tol&&k<Nmax){const x1=matVec(T,x).map((v,i)=>v+C[i]);E=norm2(vecSub(x1,x));iters.push([k+1,E,...x1]);x=x1;k++;}return{x,iters,T,C,rho};}
function gaussSeidel(A,b,x0,tol,Nmax){const n=A.length;for(let i=0;i<n;i++)if(Math.abs(A[i][i])<1e-14)throw Error('Gauss-Seidel: zero on diagonal');const D=eye(n).map((r,i)=>r.map((_,j)=>i===j?A[i][i]:0));const L=zeros(n,n),U=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){if(i>j)L[i][j]=-A[i][j];else if(i<j)U[i][j]=-A[i][j];}const DL=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)DL[i][j]=(i===j?D[i][i]:0)-L[i][j];const DLinv=inv(DL),T=matMul(DLinv,U),C=matVec(DLinv,b),rho=spectralRadius(T);const iters=[];let x=x0.slice(),E=Infinity,k=0;while(E>tol&&k<Nmax){const x1=matVec(T,x).map((v,i)=>v+C[i]);E=norm2(vecSub(x1,x));iters.push([k+1,E,...x1]);x=x1;k++;}return{x,iters,T,C,rho};}
function sor(A,b,x0,w,tol,Nmax){if(!(w>0&&w<2))throw new Recommendation('SOR requires 0<w<2');const n=A.length;for(let i=0;i<n;i++)if(Math.abs(A[i][i])<1e-14)throw Error('SOR: zero on diagonal');const D=eye(n).map((r,i)=>r.map((_,j)=>i===j?A[i][i]:0));const L=zeros(n,n),U=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){if(i>j)L[i][j]=-A[i][j];else if(i<j)U[i][j]=-A[i][j];}const DL=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)DL[i][j]=(i===j?D[i][i]:0)-w*L[i][j];const DLinv=inv(DL);const S=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)S[i][j]=(1-w)*(i===j?D[i][i]:0)+w*U[i][j];const T=matMul(DLinv,S),C=matVec(DLinv,b).map(v=>w*v),rho=spectralRadius(T);const iters=[];let x=x0.slice(),E=Infinity,k=0;while(E>tol&&k<Nmax){const x1=matVec(T,x).map((v,i)=>v+C[i]);E=norm2(vecSub(x1,x));iters.push([k+1,E,...x1]);x=x1;k++;}return{x,iters,T,C,rho};}

/* ============ Interpolation ============ */
function solveRef(A,b){const n=A.length,M=copy(A),P=eye(n),L=eye(n);for(let i=0;i<n-1;i++){let piv=i,mv=Math.abs(M[i][i]);for(let r=i+1;r<n;r++)if(Math.abs(M[r][i])>mv){mv=Math.abs(M[r][i]);piv=r;}if(piv!==i){[M[i],M[piv]]=[M[piv],M[i]];[P[i],P[piv]]=[P[piv],P[i]];for(let k=0;k<i;k++)[L[i][k],L[piv][k]]=[L[piv][k],L[i][k]];}for(let j=i+1;j<n;j++){const m=M[j][i]/M[i][i];L[j][i]=m;for(let k=i;k<n;k++)M[j][k]-=m*M[i][k];}}const y=fwdSub(L,matVec(P,b));return bwdSub(M,y);}
function vandermondeCoef(X,Y){const n=X.length,A=zeros(n,n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)A[i][j]=Math.pow(X[i],n-1-j);return solveRef(A,Y);}
function newtonDivDif(X,Y){const n=X.length,D=Array.from({length:n},()=>Array(n).fill(0));for(let i=0;i<n;i++)D[i][0]=Y[i];for(let j=1;j<n;j++)for(let i=j;i<n;i++){const num=D[i][j-1]-D[i-1][j-1];const den=X[i]-X[i-j];D[i][j]=num/den;}const coef=Array(n).fill(0).map((_,i)=>D[i][i]);return{D,coef};}
function polyMul(a,b){const r=Array(a.length+b.length-1).fill(0);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)r[i+j]+=a[i]*b[j];return r;}
function scalePoly(p,s){return p.map(c=>c*s);}
function lagrangePolys(X,Y){const n=X.length;const L=Array(n).fill(0).map(()=>[1]);for(let i=0;i<n;i++){let poly=[1];for(let j=0;j<n;j++){if(i===j)continue;poly=polyMul(poly,[1,-X[j]]);const denom=X[i]-X[j];poly=poly.map(c=>c/denom);}L[i]=poly;}const deg=Math.max(...L.map(p=>p.length))-1;const Coef=Array(deg+1).fill(0);for(let i=0;i<n;i++){const p=scalePoly(L[i],Y[i]);for(let j=0;j<p.length;j++)Coef[Coef.length-p.length+j]+=p[j];}return{L,Coef};}
function splineLineal(X,Y){const n=X.length,m=2*(n-1),A=zeros(m,m),b=Array(m).fill(0);let row=0;A[row][0]=X[0];A[row][1]=1;b[row]=Y[0];row++;for(let i=1;i<n;i++){A[row][2*i-1]=X[i];A[row][2*i]=1;b[row]=Y[i];row++;}for(let i=1;i<n-1;i++){A[row][2*i-1]=X[i];A[row][2*i]=1;A[row][2*i+1]=-X[i];A[row][2*i+2]=-1;b[row]=0;row++;}const c=solveRef(A,b);const S=[];for(let i=0;i<n-1;i++)S.push([c[2*i],c[2*i+1]]);return S;}
function splineCuadratico(X,Y){const n=X.length,m=3*(n-1),A=zeros(m,m),b=Array(m).fill(0);let row=0;A[row][0]=1;b[row]=0;row++;A[row][0]=X[0]**2;A[row][1]=X[0];A[row][2]=1;b[row]=Y[0];row++;const s0=(Y[1]-Y[0])/(X[1]-X[0]);A[row][0]=2*X[0];A[row][1]=1;A[row][2]=0;b[row]=s0;row++;for(let i=0;i<n-1;i++){const base=3*i;A[row][base]=X[i]**2;A[row][base+1]=X[i];A[row][base+2]=1;b[row]=Y[i];row++;A[row][base]=X[i+1]**2;A[row][base+1]=X[i+1];A[row][base+2]=1;b[row]=Y[i+1];row++;}for(let i=1;i<n-1;i++){const left=3*(i-1),right=3*i;A[row][left]=2*X[i];A[row][left+1]=1;A[row][right]+= -2*X[i];A[row][right+1]+= -1;b[row]=0;row++;}const Ared=A.slice(0,m).map(r=>r.slice(0,m)),bred=b.slice(0,m);const c=solveRef(Ared,bred);const S=[];for(let i=0;i<n-1;i++){const a=c[3*i],d=c[3*i+1],e=c[3*i+2];S.push([a,d,e]);}return S;}
function splineCubico(X,Y){const n=X.length,h=[],alpha=Array(n).fill(0);for(let i=0;i<n-1;i++)h[i]=X[i+1]-X[i];for(let i=1;i<n-1;i++)alpha[i]=3*((Y[i+1]-Y[i])/h[i]-(Y[i]-Y[i-1])/h[i-1]);const l=Array(n).fill(0),mu=Array(n).fill(0),z=Array(n).fill(0);l[0]=1;mu[0]=0;z[0]=0;for(let i=1;i<n-1;i++){l[i]=2*(X[i+1]-X[i-1])-h[i-1]*mu[i-1];mu[i]=h[i]/l[i];z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i];}l[n-1]=1;z[n-1]=0;const c=Array(n).fill(0),bcoef=Array(n-1).fill(0),dcoef=Array(n-1).fill(0);for(let j=n-2;j>=0;j--){c[j]=z[j]-mu[j]*c[j+1];bcoef[j]=(Y[j+1]-Y[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3;dcoef[j]=(c[j+1]-c[j])/(3*h[j]);}const S=[];for(let i=0;i<n-1;i++){const A=dcoef[i],B=c[i]-3*dcoef[i]*X[i],C=bcoef[i]-2*c[i]*X[i]+3*dcoef[i]*X[i]*X[i],D=Y[i]-bcoef[i]*X[i]+c[i]*X[i]*X[i]-dcoef[i]*X[i]*X[i]*X[i];S.push([A,B,C,D]);}return S;}

/* ============ UI bootstrap ============ */
(function UI() {
  // Map methods to workspaces with lightweight metadata describing inputs.
  const REGISTRY = {
    root: [
      { id:'INC',  name:'Incremental Search',  inputs:['fx','a','delta','nmax'] },
      { id:'BISE', name:'Bisection',           inputs:['fx','a','b','tol','imax'] },
      { id:'FALS', name:'False Position',      inputs:['fx','a','b','tol','imax'] },
      { id:'FIXP', name:'Fixed Point',         inputs:['fx','gx','a','b','x0','tol','imax'] },
      { id:'NEWT', name:'Newton',              inputs:['fx','x0','tol','imax'] },
      { id:'SECA', name:'Secant',              inputs:['fx','a','b','x0','x1','tol','imax'] },
    ],
    direct: [
      { id:'LUS',  name:'LU (simple)',         inputs:['A','b'] },
      { id:'LUP',  name:'LU (partial pivot)',  inputs:['A','b'] },
      { id:'CROT', name:'Crout',               inputs:['A','b'] },
      { id:'DOOL', name:'Doolittle',           inputs:['A','b'] },
      { id:'CHOL', name:'Cholesky (course)',   inputs:['A','b'] },
    ],
    iter: [
      { id:'JACO', name:'Jacobi',              inputs:['A','b','x0','tol','imax'] },
      { id:'GS',   name:'Gauss–Seidel',        inputs:['A','b','x0','tol','imax'] },
      { id:'SOR',  name:'SOR',                 inputs:['A','b','x0','w','tol','imax'] },
    ],
    interp: [
      { id:'VAND', name:'Vandermonde',         inputs:['pairs'] },
      { id:'NEWD', name:'Newton (div. diff.)', inputs:['pairs'] },
      { id:'LAGR', name:'Lagrange',            inputs:['pairs'] },
      { id:'SPL1', name:'Linear splines',      inputs:['pairs'] },
      { id:'SPL2', name:'Quadratic splines',   inputs:['pairs'] },
      { id:'SPL3', name:'Cubic splines',       inputs:['pairs'] },
    ]
  };

  // Shortcuts
  const $ = id => document.getElementById(id);
  const htmlEsc = s => String(s).replace(/[&<>"]/g,c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));

  // Cache DOM elements
  els.inputs = $('inputs');
  els.outputs = $('outputs');
  els.iters   = $('iters');
  els.status  = $('status');
  els.plot    = $('plot');
  els.method  = $('method');

  // Tabs
  const tabs = {
    root: $('tabRoot'),
    direct: $('tabDirect'),
    iter: $('tabIter'),
    interp: $('tabInterp')
  };

  Object.entries(tabs).forEach(([ws,btn])=>{
    btn.addEventListener('click', ()=> setWorkspace(ws));
  });

  // Populate method list for a workspace
  function setWorkspace(ws){
    state.workspace = ws;
    Object.entries(tabs).forEach(([k,btn])=>{
      btn.classList.toggle('tab-on', k===ws);
      btn.classList.toggle('tab-off', k!==ws);
    });
    populateMethods();
    renderGuide(); // show per-method guidance placeholder
    clearUI();
  }

  function populateMethods(){
    const list = REGISTRY[state.workspace];
    els.method.innerHTML = list.map(m=>`<option value="${m.id}">${m.name}</option>`).join('');
    state.methodId = list[0]?.id || null;
    renderInputsFor(state.methodId);
  }

  // Render input controls (simple, consistent templates)
  function renderInputsFor(id){
    if(!id) return;
    const meta = REGISTRY[state.workspace].find(m=>m.id===id);
    if(!meta) return;
    state.methodId = id;

    // Default examples to allow one-click Run
    const defaults = {
      fx:   'x**3 - 7*x + 6',
      gx:   '0.5*(x + 6/(x*x))',
      a:    '3',
      b:    '4',
      delta:'0.25',
      nmax: '40',
      tol:  '1e-6',
      imax: '50',
      x0:   '3.5',
      x1:   '3.8',
      w:    '1.2',
      A:    '4 -1 0\n-1 4 -1\n0 -1 3',
      bvec: '2 6 2',
      xvec: '0 0 0',
      pairs: '0 1\n1 2\n2 0\n3 2'
    };

    // Build control rows
    const ctl = [];
    meta.inputs.forEach(k=>{
      if(k==='fx') ctl.push(row('f(x) =','fxInput', defaults.fx));
      else if(k==='gx') ctl.push(row('g(x) =','gxInput', defaults.gx));
      else if(k==='a') ctl.push(row('a (left)','aInput', defaults.a));
      else if(k==='b'){
        if(state.workspace==='root') ctl.push(row('b (right)','bInput', defaults.b));
        else ctl.push(area('b (vector)','bInput', defaults.bvec,3));
      }
      else if(k==='delta') ctl.push(row('Δ (step)','deltaInput', defaults.delta));
      else if(k==='nmax') ctl.push(row('n max','nmaxInput', defaults.nmax));
      else if(k==='tol') ctl.push(row('tolerance','tolInput', defaults.tol));
      else if(k==='imax') ctl.push(row('max iterations','imaxInput', defaults.imax));
      else if(k==='x0'){
        if(state.workspace==='iter') ctl.push(area('x0 (vector)','x0Input', defaults.xvec,3));
        else ctl.push(row('x0','x0Input', defaults.x0));
      }
      else if(k==='x1') ctl.push(row('x1','x1Input', defaults.x1));
      else if(k==='w')  ctl.push(row('ω (0<w<2)','wInput', defaults.w));
      else if(k==='A')  ctl.push(area('A (matrix)', 'AInput', defaults.A,5));
      else if(k==='pairs') ctl.push(area('x;y pairs (one per line)','pairsInput', defaults.pairs,5));
    });

    els.inputs.innerHTML = `<div class="grid-3">${ctl.join('')}</div>`;
    function row(label,id,val){ return `
      <div><label class="block text-sm font-semibold mb-1">${htmlEsc(label)}</label>
      <input id="${id}" class="w-full rounded-lg border p-2" value="${htmlEsc(val)}"></div>`; }
    function area(label,id,val,rows=5){ return `
      <div class="sm:col-span-2"><label class="block text-sm font-semibold mb-1">${htmlEsc(label)}</label>
      <textarea id="${id}" rows="${rows}" class="w-full rounded-lg border p-2">${htmlEsc(val)}</textarea></div>`; }

    // update guide for this method
    renderGuide();
  }

  // Change handler for the select
  els.method.addEventListener('change', e => renderInputsFor(e.target.value));

  // Run + Clear
  $('runBtn').addEventListener('click', runCurrent);
  $('clearBtn').addEventListener('click', clearUI);
  $('clearHistory').addEventListener('click', ()=>{ localStorage.removeItem('nm_history'); renderHistory(); });

  function clearUI(){
    els.outputs.innerHTML = '';
    els.iters.innerHTML   = '';
    els.status.innerHTML  = '';
    Plots.blank(els.plot, 'Pick a method and click Run');
  }

  // Per-method guidance (short, friendly)
  function renderGuide(){
    const ws = state.workspace, id = state.methodId;
    const help = {
      root: 'Provide f(x) and a,b. The tool auto-orders [a,b] and checks sign change when required. Try Incremental Search if Bisection/False Position report “no sign change”.',
      direct: 'Paste A and b. We’ll compute L/U (and P if pivoting). Avoid singular matrices (det≈0).',
      iter: 'Provide A,b and an initial guess x0. We show the spectral radius ρ(T); if ρ>1, expect divergence.',
      interp: 'Paste data as x;y per line. Pick a scheme (Vandermonde, Newton, Lagrange, or splines) and compare the fit.'
    };
    $('methodGuide').innerHTML = `<h4>Tips</h4><p class="text-slate-600">${help[ws]}</p>`;
  }

  // History helpers
  function pushHistory(entry){
    const H = JSON.parse(localStorage.getItem('nm_history')||'[]');
    H.unshift({...entry, t: new Date().toISOString()});
    while(H.length>20) H.pop();
    localStorage.setItem('nm_history', JSON.stringify(H));
    renderHistory();
  }
  function renderHistory(){
    const H = JSON.parse(localStorage.getItem('nm_history')||'[]');
    if(!H.length){ $('history').innerHTML='<p class="text-slate-500 text-sm">No runs yet.</p>'; return; }
    $('history').innerHTML = H.map(h=>`
      <div class="rounded-lg border p-3 mb-2">
        <div class="text-xs text-slate-500">${new Date(h.t).toLocaleString()}</div>
        <div class="font-semibold">${htmlEsc(h.title)}</div>
        ${h.fn ? `<div class="text-xs text-slate-600">f(x)= <span class="code">${htmlEsc(h.fn)}</span></div>`:''}
        <div class="text-xs text-slate-600">Result: ${htmlEsc(h.result)}</div>
      </div>`).join('');
  }

  // Core runner
  function runCurrent(){
    try{
      const ws = state.workspace, id = state.methodId;
      let fnSrc=null;

      if(ws==='root'){
        const {src,f} = parseFun('fxInput'); fnSrc = src;
        if(id==='INC'){
          const a = parseNum('aInput','a'), delta = parseNum('deltaInput','Δ'), nmax = parseNum('nmaxInput','n max');
          const out = incSearch(f,a,delta,nmax);
          statusBadge(out.bracket?'ok':'warn', out.msg);
          const stepEnd = a + (delta||1)*nmax;
          const plotRange = [Math.min(a,stepEnd), Math.max(a,stepEnd)];
          Plots.func(els.plot,f,plotRange, out.bracket ? out.bracket : []);
          els.outputs.textContent = out.bracket ? `Bracket: [${fmt(out.bracket[0])}, ${fmt(out.bracket[1])}]` : 'No bracket found.';
          els.iters.innerHTML = '';
          pushHistory({title:'Incremental Search', fn: fnSrc, result: out.bracket?`[${fmt(out.bracket[0])}, ${fmt(out.bracket[1])}]`:'no bracket'});
          return;
        }
        if(id==='BISE'){
          const a = parseNum('aInput','a'), b = parseNum('bInput','b');
          const tol = parseNum('tolInput','tolerance');
          const imax = parseNum('imaxInput','max iterations');
          const r=bisection(f,a,b,tol,imax);
          statusBadge('ok','Bisection ran successfully.');
          Plots.func(els.plot,f,[Math.min(a,b),Math.max(a,b)],[r.root]);
          renderRootOutputs('Bisection',fnSrc,r);
          return;
        }
        if(id==='FALS'){
          const a = parseNum('aInput','a'), b = parseNum('bInput','b');
          const tol = parseNum('tolInput','tolerance');
          const imax = parseNum('imaxInput','max iterations');
          const r=falsePosition(f,a,b,tol,imax);
          statusBadge('ok','False position ran successfully.');
          Plots.func(els.plot,f,[Math.min(a,b),Math.max(a,b)],[r.root]);
          renderRootOutputs('False Position',fnSrc,r);
          return;
        }
        if(id==='FIXP'){
          const {f:gx}=parseFun('gxInput');
          const a = parseNum('aInput','a'), b = parseNum('bInput','b');
          const x0 = parseNum('x0Input','x0');
          const tol = parseNum('tolInput','tolerance');
          const imax = parseNum('imaxInput','max iterations');
          const r=fixedPoint(gx,[Math.min(a,b),Math.max(a,b)],x0,tol,imax);
          statusBadge('ok','Fixed point ran.');
          Plots.func(els.plot,f,[Math.min(a,b),Math.max(a,b)], r.iters.map(v=>v[2]));
          renderSeqOutputs('Fixed Point',fnSrc,r,['k','x_k','x_{k+1}','|Δ|']);
          return;
        }
        if(id==='NEWT'){
          const x0 = parseNum('x0Input','x0');
          const tol = parseNum('tolInput','tolerance');
          const imax = parseNum('imaxInput','max iterations');
          const r=newtonRoot(f,x0,tol,imax);
          statusBadge('ok','Newton ran.');
          const span = Math.max(5, Math.abs(x0)+1);
          Plots.func(els.plot,f,[x0-span,x0+span],[r.root]);
          renderSeqOutputs('Newton',fnSrc,r,['k','x_k','f(x_k)','f’(x_k)','x_{k+1}','|Δ|']);
          return;
        }
        if(id==='SECA'){
          const a = parseNum('aInput','a'), b = parseNum('bInput','b');
          const x0 = parseNum('x0Input','x0');
          const x1 = parseNum('x1Input','x1');
          const tol = parseNum('tolInput','tolerance');
          const imax = parseNum('imaxInput','max iterations');
          const r=secant(f,[Math.min(a,b),Math.max(a,b)],x0,x1,tol,imax);
          statusBadge('ok','Secant ran.');
          Plots.func(els.plot,f,[Math.min(a,b),Math.max(a,b)],[r.root]);
          renderSeqOutputs('Secant',fnSrc,r,['k','x_{k-1}','x_k','x_{k+1}','f(x_{k+1})','|Δ|']);
          return;
        }
      }

      if(ws==='direct'){
        const A = parseMat('AInput','A');
        const b = parseVec('bInput','b');
        if(state.methodId==='LUS'){ const {L,U}=luSimple(A); showLU('LU (simple)',A,b,L,U); return; }
        if(state.methodId==='LUP'){ const {L,U,P}=luPartialPivot(A); showLU('LU (partial pivot)',A,b,L,U,P); return; }
        if(state.methodId==='CROT'){ const {L,U}=crout(A); showLU('Crout',A,b,L,U); return; }
        if(state.methodId==='DOOL'){ const {L,U}=doolittle(A); showLU('Doolittle',A,b,L,U); return; }
        if(state.methodId==='CHOL'){ const {L,U}=choleskyCourse(A); showLU('Cholesky (course)',A,b,L,U); return; }
      }

      if(ws==='iter'){
        const A=parseMat('AInput','A');
        const b=parseVec('bInput','b');
        const rawX0 = ($('x0Input')?.value || '').trim();
        const x0 = rawX0 ? parseVec('x0Input','x0') : Array(A.length).fill(0);
        const tol=parseNum('tolInput','tolerance');
        const imax=parseNum('imaxInput','max iterations');
        if(state.methodId==='JACO'){ const r=jacobi(A,b,x0,tol,imax); statusBadge('ok',`Jacobi ran (ρ=${fmt(r.rho,3)}).`); renderIterSys('Jacobi',r); return; }
        if(state.methodId==='GS'){ const r=gaussSeidel(A,b,x0,tol,imax); statusBadge('ok',`Gauss–Seidel ran (ρ=${fmt(r.rho,3)}).`); renderIterSys('Gauss–Seidel',r); return; }
        if(state.methodId==='SOR'){ const w=parseNum('wInput','ω'); const r=sor(A,b,x0,w,tol,imax); statusBadge('ok',`SOR ran (ρ=${fmt(r.rho,3)}).`); renderIterSys('SOR',r); return; }
      }

      if(ws==='interp'){
        const {X,Y}=parsePairs('pairsInput','pairs');
        if(state.methodId==='VAND'){ const c=vandermondeCoef(X,Y); statusBadge('ok','Vandermonde coefficients computed.'); els.outputs.textContent='Coefficients:\n'+fixedWidthMatrix([c]); Plots.scatterFit(els.plot,{X,Y},c); pushHistory({title:'Vandermonde', result:`deg ${c.length-1}`}); els.iters.innerHTML=''; return; }
        if(state.methodId==='NEWD'){ const {D,coef}=newtonDivDif(X,Y); statusBadge('ok','Newton divided differences computed.'); els.outputs.textContent='Diagonal coef:\n'+fixedWidthMatrix([coef]); Plots.scatterFit(els.plot,{X,Y}); pushHistory({title:'Newton (div. diff.)', result:`deg ${coef.length-1}`}); els.iters.innerHTML=''; return; }
        if(state.methodId==='LAGR'){ const {Coef}=lagrangePolys(X,Y); statusBadge('ok','Lagrange polynomial computed.'); els.outputs.textContent='Coefficients:\n'+fixedWidthMatrix([Coef]); Plots.scatterFit(els.plot,{X,Y},Coef); pushHistory({title:'Lagrange', result:`deg ${Coef.length-1}`}); els.iters.innerHTML=''; return; }
        if(state.methodId==='SPL1'){ const S=splineLineal(X,Y); statusBadge('ok','Linear splines computed.'); els.outputs.textContent='Segments [p, q]:\n'+fixedWidthMatrix(S); Plots.scatterFit(els.plot,{X,Y},null,S); pushHistory({title:'Linear splines', result:`${S.length} segments`}); els.iters.innerHTML=''; return; }
        if(state.methodId==='SPL2'){ const S=splineCuadratico(X,Y); statusBadge('ok','Quadratic splines computed.'); els.outputs.textContent='Segments [A, B, C]:\n'+fixedWidthMatrix(S); Plots.scatterFit(els.plot,{X,Y},null,S); pushHistory({title:'Quadratic splines', result:`${S.length} segments`}); els.iters.innerHTML=''; return; }
        if(state.methodId==='SPL3'){ const S=splineCubico(X,Y); statusBadge('ok','Cubic splines computed.'); els.outputs.textContent='Segments [A, B, C, D]:\n'+fixedWidthMatrix(S); Plots.scatterFit(els.plot,{X,Y},null,S); pushHistory({title:'Cubic splines', result:`${S.length} segments`}); els.iters.innerHTML=''; return; }
      }
    } catch(e){
      if(e instanceof Recommendation){ statusBadge('warn', e.message); }
      else { statusBadge('err', e.message); }
      els.outputs.textContent = '';
      els.iters.innerHTML = '';
      Plots.blank(els.plot,'');
    }
  }

  function renderRootOutputs(name, fnSrc, r){
    const {root,iters,range} = r;
    const [L,R] = range || [];
    els.outputs.innerHTML = `Root ≈ ${fmt(root)}${L!==undefined?`  on [${fmt(L)}, ${fmt(R)}]`:''}`;
    renderIters(iters, ['k','L','R','mid','f(L)','f(R)','f(mid)','err']);
    pushHistory({title:name, fn: fnSrc, result:`root≈${fmt(root)}`});
  }
  function renderSeqOutputs(name, fnSrc, r, headers){
    els.outputs.textContent = `Root ≈ ${fmt(r.root)}`;
    renderIters(r.iters, headers);
    pushHistory({title:name, fn: fnSrc, result:`root≈${fmt(r.root)}`});
  }
  function renderIterSys(name, r){
    const lastErr = r.iters.at(-1)?.[1] ?? 0;
    els.outputs.textContent = `${name} finished. ρ(T)=${fmt(r.rho,3)}  ||e||=${fmt(lastErr,6)}`;
    const cols = r.iters[0]?.length || 2;
    const headers = ['k','||e||'];
    for(let i=2;i<cols;i++){ headers.push(`x${i-1}`); }
    const rows = r.iters.map(row=>[row[0],row[1],...row.slice(2)]);
    renderIters(rows, headers);
    if(r.iters.length){
      Plots.series(els.plot, r.iters.map(row=>row[1]), 'Error per iteration');
    } else {
      Plots.blank(els.plot, 'No iterations to plot');
    }
    pushHistory({title:name, result:`ρ=${fmt(r.rho,3)}`});
  }
  function renderIters(rows, headers){
    if(!rows?.length){ els.iters.innerHTML=''; return; }
    const head = `<tr>${headers.map(h=>`<th class="text-left">${h}</th>`).join('')}</tr>`;
    const first3 = rows.slice(0,3);
    const last3  = rows.slice(-3);
    const bodyRows = rows.length>6 ? [...first3, [ '…', ...Array(headers.length-1).fill('…') ], ...last3] : rows;
    const body = bodyRows.map(r=>`<tr>${r.map(c=>`<td class="code w-ch">${htmlEsc(fmt(c))}</td>`).join('')}</tr>`).join('');
    els.iters.innerHTML = `<table class="w-full table-sm"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  function showLU(title,A,b,L,U,P){
    const y = fwdSub(L, P ? matVec(P,b) : b);
    const x = bwdSub(U, y);
    statusBadge('ok', `${title} ran successfully.`);
    els.outputs.textContent = `${title}\n\nL:\n${fixedWidthMatrix(L)}\n\nU:\n${fixedWidthMatrix(U)}\n\nx:\n${fixedWidthMatrix([x])}`;
    els.iters.innerHTML = '';
    Plots.blank(els.plot, 'Factorization – no plot');
    pushHistory({title, result:`||x||≈${fmt(norm2(x))}`});
  }

  renderHistory();
  // Ensure the page is interactive immediately
  setWorkspace('root');
})();

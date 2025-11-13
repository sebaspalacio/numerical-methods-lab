/* ============ Small helpers ============ */
const fmt=(x,k=6)=>Number.isFinite(x)?Number(x).toFixed(k):String(x);
const zeros=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));
const eye=n=>{const I=zeros(n,n);for(let i=0;i<n;i++)I[i][i]=1;return I;};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const matVec=(A,x)=>A.map(r=>dot(r,x));
const vecSub=(a,b)=>a.map((v,i)=>v-b[i]);
const norm2=v=>Math.sqrt(dot(v,v));
const copy=A=>A.map(r=>r.slice());
class Recommendation extends Error{}
function fixedWidthMatrix(M,k=6){
  const s=M.map(r=>r.map(v=>fmt(v,k)));
  const w=Math.max(...s.flat().map(t=>t.length));
  return s.map(r=>' '+r.map(t=>t.padStart(w,' ')).join(' ')+' ').join('\\n');
}
function statusBadge(kind,msg){
  const cls=kind==='ok'?'badge-ok':kind==='warn'?'badge-warn':'badge-err';
  document.getElementById('status').innerHTML =
    '<div class="card"><span class="badge '+cls+'">'+kind.toUpperCase()+
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
  const allowed=/[0-9a-zA-Z_\\s+*\\/^.(),\\-]/;
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

/* ============ Workspaces & Methods ============ */
const METHODS_ROOT={
  'Incremental Search':{
    id:'incremental',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',hint:'e.g., x**3 - 7*x + 6',def:'x**3 - 7*x + 6'},
      {id:'a',label:'a (interval)',type:'num',def:'3'},
      {id:'delta',label:'Δ (step)',type:'num',def:'-0.5'},
      {id:'nmax',label:'Max steps',type:'num',def:'100'}
    ],
    prereq:'Scans from a with step Δ to locate a sign change.',
    run:()=>{
      const {src,f}=parseFun('fx'); const a=parseNum('a'), d=parseNum('delta'), n=parseNum('nmax');
      const {bracket,msg}=incSearch(f,a,d,n);
      if(!bracket) throw new Recommendation(msg+' Try another Δ or start point.');
      return {status:['ok','Bracket found'], summary:`Use [${fmt(bracket[0])}, ${fmt(bracket[1])}]`, details:{fsrc:src, plot:(div)=>Plots.func(div,f,bracket,[bracket[0],bracket[1]])}};
    }
  },
  'Bisection':{
    id:'bisection',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',def:'x**3 - 7*x + 6'},
      {id:'a',label:'a',type:'num',def:'3'},
      {id:'b',label:'b',type:'num',def:'0.5'},
      {id:'tol',label:'tol',type:'num',def:'1e-6'},
      {id:'kmax',label:'max iterations',type:'num',def:'100'}
    ],
    prereq:'Requires sign change on [min(a,b), max(a,b)].',
    run:()=>{const {src,f}=parseFun('fx');const a=parseNum('a'),b=parseNum('b'),tol=parseNum('tol'),k=parseNum('kmax');const out=bisection(f,a,b,tol,k);return {status:['ok','Bisection completed'], summary:`root ≈ ${fmt(out.root)}`, details:{iters:out.iters,fsrc:src, plot:(div)=>Plots.func(div,f,[Math.min(a,b),Math.max(a,b)],[out.root])}};}
  },
  'False Position':{
    id:'falsePosition',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',def:'x**3 - 7*x + 6'},
      {id:'a',label:'a',type:'num',def:'3'},
      {id:'b',label:'b',type:'num',def:'0.5'},
      {id:'tol',label:'tol',type:'num',def:'1e-6'},
      {id:'kmax',label:'max iterations',type:'num',def:'100'}
    ],
    prereq:'Requires sign change on [a,b].',
    run:()=>{const {src,f}=parseFun('fx');const a=parseNum('a'),b=parseNum('b'),tol=parseNum('tol'),k=parseNum('kmax');const out=falsePosition(f,a,b,tol,k);return {status:['ok','False Position completed'], summary:`root ≈ ${fmt(out.root)}`, details:{iters:out.iters,fsrc:src, plot:(div)=>Plots.func(div,f,[Math.min(a,b),Math.max(a,b)],[out.root])}};}
  },
  'Fixed Point':{
    id:'fixedPoint',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',def:'x**3 - 7*x + 6'},
      {id:'gx',label:'g(x) =',type:'fun',hint:'e.g., (x + (7*x - 6)/3)/2',def:'(x + (7*x - 6)/3)/2'},
      {id:'a',label:'a',type:'num',def:'3'},
      {id:'b',label:'b',type:'num',def:'0.5'},
      {id:'x0',label:'x0',type:'num',def:'1.5'},
      {id:'tol',label:'tol',type:'num',def:'1e-6'},
      {id:'kmax',label:'max iterations',type:'num',def:'100'}
    ],
    prereq:'Prefer a contractive g and invariant interval.',
    run:()=>{const {src,f}=parseFun('fx');const {f:gx}=parseFun('gx');const a=parseNum('a'),b=parseNum('b'),x0=parseNum('x0'),tol=parseNum('tol'),k=parseNum('kmax');const out=fixedPoint(gx,[Math.min(a,b),Math.max(a,b)],x0,tol,k);return {status:['ok','Fixed Point completed'], summary:`root ≈ ${fmt(out.root)}`, details:{iters:out.iters,fsrc:src, plot:(div)=>Plots.func(div,f,[Math.min(a,b),Math.max(a,b)],out.iters.map(r=>r[2]))}};}
  },
  'Newton':{
    id:'newton',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',def:'x**3 - 7*x + 6'},
      {id:'x0',label:'x0',type:'num',def:'0'},
      {id:'tol',label:'tol',type:'num',def:'1e-6'},
      {id:'kmax',label:'max iterations',type:'num',def:'50'}
    ],
    prereq:'Avoid points where the derivative is near zero.',
    run:()=>{const {src,f}=parseFun('fx');const x0=parseNum('x0'),tol=parseNum('tol'),k=parseNum('kmax');const out=newtonRoot(f,x0,tol,k);return {status:['ok','Newton completed'], summary:`root ≈ ${fmt(out.root)}`, details:{iters:out.iters,fsrc:src, plot:(div)=>Plots.func(div,f,[x0-5,x0+5],out.iters.map(r=>r[4]))}};}
  },
  'Secant':{
    id:'secant',
    inputs:[
      {id:'fx',label:'f(x) =',type:'fun',def:'x**3 - 7*x + 6'},
      {id:'a',label:'a',type:'num',def:'3'},
      {id:'b',label:'b',type:'num',def:'0.5'},
      {id:'x0',label:'x0',type:'num',def:'1'},
      {id:'x1',label:'x1',type:'num',def:'2'},
      {id:'tol',label:'tol',type:'num',def:'1e-6'},
      {id:'kmax',label:'max iterations',type:'num',def:'100'}
    ],
    prereq:'Denominator must not be near zero.',
    run:()=>{const {src,f}=parseFun('fx');const a=parseNum('a'),b=parseNum('b'),x0=parseNum('x0'),x1=parseNum('x1'),tol=parseNum('tol'),k=parseNum('kmax');const out=secant(f,[Math.min(a,b),Math.max(a,b)],x0,x1,tol,k);return {status:['ok','Secant completed'], summary:`root ≈ ${fmt(out.root)}`, details:{iters:out.iters,fsrc:src, plot:(div)=>Plots.func(div,f,[Math.min(a,b),Math.max(a,b)],out.iters.map(r=>r[3]))}};}
  }
};

/* Matrix input (paste or build) */
function renderMatrixWidget(targetId,labelText,defText=''){
  const wrap=document.createElement('div');
  const lab=document.createElement('label'); lab.className='text-sm font-semibold mb-1'; lab.textContent=labelText; wrap.appendChild(lab);

  const tabs=document.createElement('div'); tabs.className='tabset mb-2';
  const btnPaste=document.createElement('button'); btnPaste.type='button'; btnPaste.className='on'; btnPaste.textContent='Paste';
  const btnBuild=document.createElement('button'); btnBuild.type='button'; btnBuild.textContent='Build';
  tabs.append(btnPaste,btnBuild); wrap.appendChild(tabs);

  const area=document.createElement('textarea'); area.id=targetId; area.className='rounded-lg border p-2 code w-full'; area.rows=4; area.placeholder='rows by line, columns by space or comma'; area.value=defText; wrap.appendChild(area);

  const buildBox=document.createElement('div'); buildBox.style.display='none'; buildBox.className='mt-2 space-y-2';
  buildBox.innerHTML = `
    <div class="flex gap-2 items-center">
      <label class="text-sm">Rows</label><input id="${targetId}_r" type="number" min="1" max="8" value="3" class="border rounded p-1 w-20"/>
      <label class="text-sm">Cols</label><input id="${targetId}_c" type="number" min="1" max="8" value="3" class="border rounded p-1 w-20"/>
      <button type="button" id="${targetId}_mk" class="btn btn-ghost">Make grid</button>
      <button type="button" id="${targetId}_ex" class="btn btn-ghost">Fill example</button>
    </div>
    <div id="${targetId}_grid" class="space-y-1"></div>
  `;
  wrap.appendChild(buildBox);

  function gridToText(){
    const r=Number(document.getElementById(targetId+'_r').value);
    const c=Number(document.getElementById(targetId+'_c').value);
    const g=document.getElementById(targetId+'_grid');
    const lines=[];
    for(let i=0;i<r;i++){
      const row=[]; for(let j=0;j<c;j++){ row.push(Number(g.querySelector(`#${targetId}_cell_${i}_${j}`).value||'0')); }
      lines.push(row.join(' '));
    }
    area.value=lines.join('\\n');
  }
  function makeGrid(fill){
    const r=Number(document.getElementById(targetId+'_r').value);
    const c=Number(document.getElementById(targetId+'_c').value);
    const g=document.getElementById(targetId+'_grid'); g.innerHTML='';
    for(let i=0;i<r;i++){
      const row=document.createElement('div'); row.className='flex gap-1';
      for(let j=0;j<c;j++){
        const inp=document.createElement('input');
        inp.type='number'; inp.step='any'; inp.className='border rounded p-1 w-24 code'; inp.id=`${targetId}_cell_${i}_${j}`;
        inp.value = fill ? (i===j?1:0) : 0;
        inp.addEventListener('input', gridToText);
        row.appendChild(inp);
      }
      g.appendChild(row);
    }
    gridToText();
  }

  btnPaste.onclick=()=>{btnPaste.className='on';btnBuild.className='';area.style.display='block';buildBox.style.display='none';};
  btnBuild.onclick=()=>{btnBuild.className='on';btnPaste.className='';area.style.display='none';buildBox.style.display='block';makeGrid(false);};
  buildBox.querySelector('#'+targetId+'_mk').onclick=()=>makeGrid(false);
  buildBox.querySelector('#'+targetId+'_ex').onclick=()=>makeGrid(true);

  return wrap;
}

const METHODS_DIRECT={
  'LU (simple)':{id:'luSimple',inputs:[{id:'A',label:'A',type:'mat',def:'4 -1 0\\n-1 4 -1\\n0 -1 4'},{id:'b',label:'b (vector)',type:'vec',def:'2 6 2'}],prereq:'Square invertible A.',run:()=>{const A=parseMat('A'),b=parseVec('b');if(A.length!==A[0].length)throw Error('A must be square');const {L,U}=luSimple(A);const x=bwdSub(U,fwdSub(L,b));return{status:['ok','LU (simple) completed'],summary:`x = [${x.map(v=>fmt(v)).join(', ')}]`,details:{matrices:{L,U}}};}},
  'LU (partial pivot)':{id:'luPP',inputs:[{id:'A',label:'A',type:'mat',def:'0 2 9\\n1 -1 2\\n3 2 1'},{id:'b',label:'b (vector)',type:'vec',def:'1 2 3'}],prereq:'More robust than simple LU.',run:()=>{const A=parseMat('A'),b=parseVec('b');if(A.length!==A[0].length)throw Error('A must be square');const {L,U,P}=luPartialPivot(A);const x=bwdSub(U,fwdSub(L,matVec(P,b)));return{status:['ok','LU (partial pivot) completed'],summary:`x = [${x.map(v=>fmt(v)).join(', ')}]`,details:{matrices:{L,U,P}}};}},
  'Crout':{id:'crout',inputs:[{id:'A',label:'A',type:'mat',def:'10 2 3\\n3 10 4\\n3 4 10'},{id:'b',label:'b (vector)',type:'vec',def:'1 2 3'}],prereq:'L non-unit diag; U unit diag.',run:()=>{const A=parseMat('A'),b=parseVec('b');const{L,U}=crout(A);const x=bwdSub(U,fwdSub(L,b));return{status:['ok','Crout completed'],summary:`x = [${x.map(v=>fmt(v)).join(', ')}]`,details:{matrices:{L,U}}};}},
  'Doolittle':{id:'doolittle',inputs:[{id:'A',label:'A',type:'mat',def:'10 2 3\\n3 10 4\\n3 4 10'},{id:'b',label:'b (vector)',type:'vec',def:'1 2 3'}],prereq:'L unit diag; U general.',run:()=>{const A=parseMat('A'),b=parseVec('b');const{L,U}=doolittle(A);const x=bwdSub(U,fwdSub(L,b));return{status:['ok','Doolittle completed'],summary:`x = [${x.map(v=>fmt(v)).join(', ')}]`,details:{matrices:{L,U}}};}},
  'Cholesky':{id:'choleskyCourse',inputs:[{id:'A',label:'A (symmetric; SPD expected)',type:'mat',def:'4 1 1\\n1 3 0\\n1 0 2'},{id:'b',label:'b (vector)',type:'vec',def:'1 2 3'}],prereq:'Stops if a non-positive pivot appears.',run:()=>{const A=parseMat('A'),b=parseVec('b');const{L,U}=choleskyCourse(A);const x=bwdSub(U,fwdSub(L,b));return{status:['ok','Cholesky completed'],summary:`x = [${x.map(v=>fmt(v)).join(', ')}]`,details:{matrices:{L,U}}};}}
};

const METHODS_ITER={
  'Jacobi':{id:'jacobi',inputs:[{id:'A',label:'A',type:'mat',def:'4 -1 0\\n-1 4 -1\\n0 -1 4'},{id:'b',label:'b',type:'vec',def:'2 6 2'},{id:'x0',label:'x0',type:'vec',def:'0 0 0'},{id:'tol',label:'tol',type:'num',def:'1e-7'},{id:'N',label:'Nmax',type:'num',def:'100'}],prereq:'Diagonal must have no zeros. We show ρ(T).',run:()=>{const A=parseMat('A'),b=parseVec('b'),x0=parseVec('x0'),tol=parseNum('tol'),N=parseNum('N');const out=jacobi(A,b,x0,tol,N);const note=out.rho>1?['warn',`ρ(T)=${fmt(out.rho,6)} > 1 (may diverge)`]:['ok',`ρ(T)=${fmt(out.rho,6)}`];return{status:note,summary:`x ≈ [${out.x.map(v=>fmt(v)).join(', ')}]`,details:{iters:out.iters,matrices:{T:out.T,C:out.C},series:out.iters.map(r=>r[1])}};}},
  'Gauss-Seidel':{id:'gs',inputs:[{id:'A',label:'A',type:'mat',def:'4 -1 0\\n-1 4 -1\\n0 -1 4'},{id:'b',label:'b',type:'vec',def:'2 6 2'},{id:'x0',label:'x0',type:'vec',def:'0 0 0'},{id:'tol',label:'tol',type:'num',def:'1e-7'},{id:'N',label:'Nmax',type:'num',def:'100'}],prereq:'Diagonal without zeros. We show ρ(T).',run:()=>{const A=parseMat('A'),b=parseVec('b'),x0=parseVec('x0'),tol=parseNum('tol'),N=parseNum('N');const out=gaussSeidel(A,b,x0,tol,N);return{status:['ok',`ρ(T)=${fmt(out.rho,6)}`],summary:`x ≈ [${out.x.map(v=>fmt(v)).join(', ')}]`,details:{iters:out.iters,matrices:{T:out.T,C:out.C},series:out.iters.map(r=>r[1])}};}},
  'SOR (w)':{id:'sor',inputs:[{id:'A',label:'A',type:'mat',def:'4 -1 0\\n-1 4 -1\\n0 -1 4'},{id:'b',label:'b',type:'vec',def:'2 6 2'},{id:'x0',label:'x0',type:'vec',def:'0 0 0'},{id:'w',label:'w (0<w<2)',type:'num',def:'1.5'},{id:'tol',label:'tol',type:'num',def:'1e-7'},{id:'N',label:'Nmax',type:'num',def:'100'}],prereq:'Pick 0<w<2. We show ρ(T).',run:()=>{const A=parseMat('A'),b=parseVec('b'),x0=parseVec('x0'),w=parseNum('w'),tol=parseNum('tol'),N=parseNum('N');const out=sor(A,b,x0,w,tol,N);return{status:['ok',`ρ(T)=${fmt(out.rho,6)}`],summary:`x ≈ [${out.x.map(v=>fmt(v)).join(', ')}]`,details:{iters:out.iters,matrices:{T:out.T,C:out.C},series:out.iters.map(r=>r[1])}};}}
};

const METHODS_INTERP={
  'Vandermonde':{id:'vandermonde',inputs:[{id:'xy',label:'Table x;y per line',type:'pairs',def:'-1;3\\n0;2\\n1;3'}],prereq:'Solves polynomial coefficients via Vandermonde.',run:()=>{const {X,Y}=parsePairs('xy');const coef=vandermondeCoef(X,Y);return{status:['ok','Vandermonde completed'],summary:`coef = [${coef.map(c=>fmt(c)).join(', ')}]`,details:{poly:coef,pts:{X,Y}}};}},
  'Newton (divided differences)':{id:'newtonDD',inputs:[{id:'xy',label:'Table x;y per line',type:'pairs',def:'-1;3\\n0;2\\n1;3'}],prereq:'Builds the divided-difference table and coefficients.',run:()=>{const {X,Y}=parsePairs('xy');const {D,coef}=newtonDivDif(X,Y);return{status:['ok','Newton (divided differences) completed'],summary:`b = [${coef.map(c=>fmt(c)).join(', ')}]`,details:{D,pts:{X,Y}}};}},
  'Lagrange':{id:'lagrange',inputs:[{id:'xy',label:'Table x;y per line',type:'pairs',def:'-1;3\\n0;2\\n1;3'}],prereq:'Builds L_i(x) and the final P(x).',run:()=>{const {X,Y}=parsePairs('xy');const {L,Coef}=lagrangePolys(X,Y);return{status:['ok','Lagrange completed'],summary:`P(x) coefficients = [${Coef.map(c=>fmt(c)).join(', ')}]`,details:{L,pts:{X,Y},poly:Coef}};}},
  'Linear splines':{id:'splLin',inputs:[{id:'xy',label:'Table x;y per line (x increasing)',type:'pairs',def:'0;0\\n1;1\\n2;0.5\\n3;1.5'}],prereq:'Piecewise linear between points.',run:()=>{const {X,Y}=parsePairs('xy');const C=splineLineal(X,Y);return{status:['ok','Linear splines ready'],summary:`${C.length} segments`,details:{spl:C,pts:{X,Y}}};}},
  'Quadratic splines':{id:'splQuad',inputs:[{id:'xy',label:'Table x;y per line',type:'pairs',def:'0;0\\n1;1\\n2;0.5\\n3;1.5'}],prereq:'C1 continuity with a boundary choice.',run:()=>{const {X,Y}=parsePairs('xy');const C=splineCuadratico(X,Y);return{status:['ok','Quadratic splines ready'],summary:`${C.length} segments`,details:{spl:C,pts:{X,Y}}};}},
  'Cubic splines (natural)':{id:'splCub',inputs:[{id:'xy',label:'Table x;y per line',type:'pairs',def:'0;0\\n1;1\\n2;0.5\\n3;1.5'}],prereq:'Natural boundary S""(x0)=S""(xn)=0.',run:()=>{const {X,Y}=parsePairs('xy');const C=splineCubico(X,Y);return{status:['ok','Cubic splines ready'],summary:`${C.length} segments`,details:{spl:C,pts:{X,Y}}};}}
};

const METHODS_BY_WS={root:METHODS_ROOT,direct:METHODS_DIRECT,iter:METHODS_ITER,interp:METHODS_INTERP};

const METHOD_GUIDES={
  incremental:{
    title:'Incremental search',
    summary:'Scan f(x) from the starting point a with step Δ until you detect a sign change.',
    checklist:[
      'Provide f(x), the start point a, the step Δ, and the maximum number of steps.',
      'The sign of Δ controls the direction; raise “Max steps” to cover a wider interval.'
    ],
    reminders:[
      'If no sign change appears, tweak Δ or move the starting point.'
    ]
  },
  bisection:{
    title:'Bisection',
    summary:'Shrink the bracket by halving it while the endpoints keep opposite signs.',
    checklist:[
      'Enter f(x), endpoints a and b (in any order), the tolerance, and the max iterations.',
      'Before running, confirm that f(a) and f(b) have opposite signs.'
    ],
    reminders:[
      'If no sign change exists, find a new interval or try incremental search.'
    ]
  },
  falsePosition:{
    title:'False position',
    summary:'Build secants with the bracket endpoints to approximate the root.',
    checklist:[
      'Provide f(x), the interval [a,b], the tolerance, and the max iterations.',
      'Use it when you already have a sign-change bracket and want faster convergence than bisection.'
    ],
    reminders:[
      'If iterations stall at one endpoint, adjust the bracket or switch to the secant method.'
    ]
  },
  fixedPoint:{
    title:'Fixed point',
    summary:'Iterate xₖ₊₁ = g(xₖ) within a stable interval to satisfy f(x)=0.',
    checklist:[
      'Enter f(x), g(x), the interval [a,b], the seed x₀, the tolerance, and the max iterations.',
      'Ensure g(x) keeps the iterates inside the chosen interval.'
    ],
    reminders:[
      'If iterates leave the range, adjust g(x), the seed, or tighten the interval.'
    ]
  },
  newton:{
    title:'Newton-Raphson',
    summary:'Use fʼ(x) to obtain quadratic convergence near the root.',
    checklist:[
      'Define f(x), the initial guess x₀, the tolerance, and the iteration limit.',
      'Pick x₀ close to the root and avoid areas where fʼ(x) ≈ 0.'
    ],
    reminders:[
      'If the denominator nears zero, change the seed or try the secant method.'
    ]
  },
  secant:{
    title:'Secant',
    summary:'Approximate the derivative with consecutive evaluations instead of computing fʼ(x).',
    checklist:[
      'Provide f(x), a reference interval [a,b], seeds x₀ and x₁, the tolerance, and the max iterations.',
      'Keep the seeds within or near the interval for stability.'
    ],
    reminders:[
      'If iterates exit the interval or the denominator gets small, adjust the seeds.'
    ]
  },
  luSimple:{
    title:'LU (simple)',
    summary:'Factor A into L and U without pivoting; best for well-conditioned matrices.',
    checklist:[
      'Enter a square matrix A and the vector b.',
      'Switch to partial pivoting if a pivot approaches zero.'
    ],
    reminders:[
      'Check that the diagonal of A has no zeros.'
    ]
  },
  luPP:{
    title:'LU (partial pivot)',
    summary:'Swap rows when needed to obtain a more stable LU factorization.',
    checklist:[
      'Provide square A and b; the method outputs L, U, and permutation matrix P.',
      'Ideal when A has small or zero diagonal entries.'
    ],
    reminders:[
      'If A is singular the process stops; verify the data.'
    ]
  },
  crout:{
    title:'Crout',
    summary:'Produces L with free diagonal and unit-diagonal U to solve Ax=b.',
    checklist:[
      'Enter the square matrix A and the vector b.',
      'Use it when you want L to keep the scale of the pivots.'
    ],
    reminders:[
      'Ensure no diagonal pivot is zero before proceeding.'
    ]
  },
  doolittle:{
    title:'Doolittle',
    summary:'Builds L with ones on the diagonal and a general U.',
    checklist:[
      'Enter the square matrix A and the matching vector b.',
      'Similar to Crout but normalizes the diagonal of L to one.'
    ],
    reminders:[
      'If a pivot vanishes, consider partial pivoting.'
    ]
  },
  choleskyCourse:{
    title:'Cholesky',
    summary:'Factor SPD matrices into L·U (L lower, U upper).',
    checklist:[
      'Provide a symmetric positive-definite matrix A and vector b.',
      'Stops immediately if a non-positive pivot is detected.'
    ],
    reminders:[
      'If it fails, re-check the symmetry and positivity of A before retrying.'
    ]
  },
  jacobi:{
    title:'Jacobi',
    summary:'Update all variables in parallel using the previous iterate.',
    checklist:[
      'Enter A, b, the initial vector x₀, tolerance, and Nmax.',
      'Ensure the diagonal of A has no zeros and watch ρ(T) to assess convergence.'
    ],
    reminders:[
      'If ρ(T) ≥ 1, reorder the system or switch methods.'
    ]
  },
  gs:{
    title:'Gauss-Seidel',
    summary:'Reuse freshly updated values to converge faster than Jacobi.',
    checklist:[
      'Enter A, b, the initial vector x₀, tolerance, and Nmax.',
      'Works best with diagonally dominant matrices; monitor ρ(T).'
    ],
    reminders:[
      'If it diverges, try reordering rows or switch to SOR with a suitable ω.'
    ]
  },
  sor:{
    title:'SOR',
    summary:'Add a relaxation factor ω to speed up Gauss-Seidel.',
    checklist:[
      'Provide A, b, the initial vector x₀, ω (0<ω<2), tolerance, and Nmax.',
      'Start near ω≈1 and adjust based on the convergence rate.'
    ],
    reminders:[
      'An out-of-range ω triggers an error; if ρ(T) stays large, try another ω.'
    ]
  },
  vandermonde:{
    title:'Vandermonde',
    summary:'Solve the Vandermonde system to get the interpolating polynomial coefficients.',
    checklist:[
      'Enter x;y pairs (one per line) to build the matrix.',
      'Best for a few points; can become unstable with large datasets.'
    ],
    reminders:[
      'Sort points by x to interpret the resulting polynomial more easily.'
    ]
  },
  newtonDD:{
    title:'Newton (divided differences)',
    summary:'Build the divided-difference table and Newton form of the polynomial.',
    checklist:[
      'Enter ordered x;y pairs; the triangular table is created automatically.',
      'Lets you append new points while reusing previous results.'
    ],
    reminders:[
      'Avoid repeated x-values to prevent division by zero.'
    ]
  },
  lagrange:{
    title:'Lagrange',
    summary:'Sum the weighted basis polynomials Lᵢ(x) to form P(x).',
    checklist:[
      'Provide the x;y pairs; the bases and final coefficients are computed.',
      'Makes it easy to evaluate P(x) anywhere without solving extra systems.'
    ],
    reminders:[
      'With many points the polynomial may oscillate (Runge phenomenon).'
    ]
  },
  splLin:{
    title:'Linear splines',
    summary:'Connect each node pair with continuous line segments.',
    checklist:[
      'Enter x;y pairs with increasing x.',
      'Great for quick approximations without oscillations.'
    ],
    reminders:[
      'Add more nodes if you need to capture more curvature.'
    ]
  },
  splQuad:{
    title:'Quadratic splines',
    summary:'Use quadratic pieces with value and first-derivative continuity.',
    checklist:[
      'Enter the x;y pairs; a boundary condition closes the system.',
      'Provides extra smoothness over linear splines.'
    ],
    reminders:[
      'Make sure each x-value is unique.'
    ]
  },
  splCub:{
    title:'Natural cubic splines',
    summary:'Deliver a smooth interpolation with zero second derivatives at the ends.',
    checklist:[
      'Enter ordered x;y pairs to solve the tridiagonal system.',
      'Guarantees continuity in value plus first and second derivatives.'
    ],
    reminders:[
      'For different boundary conditions, consider another spline variant.'
    ]
  }
};


/* ============ UI ============ */
const els={
  tabRoot:document.getElementById('tabRoot'),
  tabDirect:document.getElementById('tabDirect'),
  tabIter:document.getElementById('tabIter'),
  tabInterp:document.getElementById('tabInterp'),
  method:document.getElementById('method'),
  runBtn:document.getElementById('runBtn'),
  clearBtn:document.getElementById('clearBtn'),
  methodGuide:document.getElementById('methodGuide'),
  inputs:document.getElementById('inputs'),
  status:document.getElementById('status'),
  plot:document.getElementById('plot'),
  outputs:document.getElementById('outputs'),
  iters:document.getElementById('iters'),
  history:document.getElementById('history'),
  clearHistory:document.getElementById('clearHistory')
};

function wireTabs(){
  els.tabRoot.addEventListener('click', ()=>setTab('root'));
  els.tabDirect.addEventListener('click', ()=>setTab('direct'));
  els.tabIter.addEventListener('click', ()=>setTab('iter'));
  els.tabInterp.addEventListener('click', ()=>setTab('interp'));
}
function setTab(ws){
  els.tabRoot.className   = 'btn ' + (ws==='root'?'tab-on':'tab-off');
  els.tabDirect.className = 'btn ' + (ws==='direct'?'tab-on':'tab-off');
  els.tabIter.className   = 'btn ' + (ws==='iter'?'tab-on':'tab-off');
  els.tabInterp.className = 'btn ' + (ws==='interp'?'tab-on':'tab-off');
  window.WS = ws;
  populateMethods();
  clearPanels();
}
function populateMethods(){
  const sel=els.method; sel.innerHTML='';
  const dict=METHODS_BY_WS[WS];
  Object.keys(dict).forEach(name=>{
    const o=document.createElement('option'); o.value=dict[name].id; o.textContent=name; sel.appendChild(o);
  });
  renderInputs();
  renderGuide();
}

function renderInputs(){
  const dict=METHODS_BY_WS[WS];
  const id  = els.method.value || Object.values(dict)[0].id;
  const meta=Object.values(dict).find(m=>m.id===id) || Object.values(dict)[0];
  els.method.value=meta.id;
  const box=els.inputs; box.innerHTML='';

  (meta.inputs||[]).forEach(inp=>{
    const w=document.createElement('div'); w.className='flex flex-col mb-3';
    // label
    if(inp.type!=='matBuildOnly'){ const lab=document.createElement('label'); lab.className='text-sm font-semibold mb-1'; lab.textContent=inp.label; w.appendChild(lab); }
    // control
    let el;
    if(inp.type==='fun'){ el=document.createElement('input'); el.placeholder=inp.hint||'e.g., x**2 + 2*x + 1'; el.value=inp.def||'x**2 + 2*x + 1'; el.id=inp.id; el.className='rounded-lg border p-2 code w-full'; w.appendChild(el); }
    else if(inp.type==='num'){ el=document.createElement('input'); el.value=inp.def||'0'; el.id=inp.id; el.className='rounded-lg border p-2 code w-full'; w.appendChild(el); }
    else if(inp.type==='vec'){ el=document.createElement('textarea'); el.rows=2; el.placeholder='e.g., 1 1 1'; el.value=inp.def||''; el.id=inp.id; el.className='rounded-lg border p-2 code'; w.appendChild(el); }
    else if(inp.type==='pairs'){ el=document.createElement('textarea'); el.rows=4; el.placeholder='x;y per line'; el.value=inp.def||''; el.id=inp.id; el.className='rounded-lg border p-2 code'; w.appendChild(el); }
    else if(inp.type==='mat'){ // dual widget
      const def=inp.def||'';
      const widget=renderMatrixWidget(inp.id, inp.label, def);
      w.appendChild(widget);
    }
    box.appendChild(w);
  });
}

function clearPanels(){ els.status.innerHTML=''; els.outputs.innerHTML=''; els.iters.innerHTML=''; els.plot.innerHTML=''; }

function renderGuide(){
  const id = els.method.value;
  const info = METHOD_GUIDES[id];
  if(!info){
    els.methodGuide.innerHTML = '<div class="text-sm text-slate-600">Select a method to see the tailored suggestions.</div>';
    return;
  }
  const checklist = info.checklist?.length
    ? '<div><h5 class="text-sm font-semibold text-slate-700">To get started</h5><ul class="list-disc pl-5 text-sm text-slate-700 space-y-1">'+info.checklist.map(item=>'<li>'+item+'</li>').join('')+'</ul></div>'
    : '';
  const reminders = info.reminders?.length
    ? '<div class="pt-2 border-t border-blue-100 mt-2"><h5 class="text-sm font-semibold text-slate-700">Quick tips</h5><ul class="list-disc pl-5 text-sm text-slate-600 space-y-1 mt-1">'+info.reminders.map(item=>'<li>'+item+'</li>').join('')+'</ul></div>'
    : '';
  els.methodGuide.innerHTML = `
    <div class="space-y-3">
      <div>
        <p class="text-xs font-semibold uppercase tracking-widest text-blue-600">Method guide</p>
        <h4 class="font-semibold text-slate-900">${info.title}</h4>
        <p class="text-sm text-slate-600 mt-1">${info.summary}</p>
      </div>
      ${checklist}
      ${reminders}
    </div>
  `;
}

/* ============ Run & History ============ */
document.getElementById('runBtn').addEventListener('click', run);
document.getElementById('clearBtn').addEventListener('click', ()=>clearPanels());
els.method.addEventListener('change', ()=>{ renderInputs(); renderGuide(); clearPanels(); });

function run(){
  try{
    const dict=METHODS_BY_WS[WS]; const id=els.method.value;
    const meta=Object.values(dict).find(m=>m.id===id); if(!meta) throw Error('Unknown method');
    const res=meta.run();

    const kind=res.status?.[0]||'ok'; statusBadge(kind,res.status?.[1]||'Done');

    const out=els.outputs; out.innerHTML='';
    if(res.details?.matrices){
      const mats=res.details.matrices;
      for(const k in mats){
        out.innerHTML += '<div><div class="font-semibold">'+k+':</div><pre class="code">'+fixedWidthMatrix(mats[k])+'</pre></div>';
      }
    }
    if(res.details?.fsrc){ out.innerHTML += '<div><span class="font-semibold">f(x):</span> <span class="code">'+res.details.fsrc+'</span></div>'; }
    out.innerHTML += '<div><span class="font-semibold">Summary:</span> '+res.summary+'</div>';

    const it=els.iters; it.innerHTML='';
    if(res.details?.iters?.length){
      const rows=res.details.iters;
      const head='<tr class="text-left">'+rows[0].map((_,i)=>'<th class="table-sm">c'+(i+1)+'</th>').join('')+'</tr>';
      const rowHtml=a=>a.map(r=>'<tr>'+r.map(v=>'<td class="table-sm">'+(typeof v==='number'?fmt(v):v)+'</td>').join('')+'</tr>').join('');
      const first=rows.slice(0,3), last=rows.slice(-3), mid=rows.length>6?'<tr><td class="table-sm text-center text-slate-400" colspan="'+rows[0].length+'">…</td></tr>':'';
      it.innerHTML='<table class="w-full table-sm">'+head+rowHtml(first)+mid+(rows.length>3?rowHtml(last):'')+'</table>';
    }

    const p=els.plot; p.innerHTML='';
    if(typeof res.details?.plot==='function'){ res.details.plot('plot'); }
    else if(WS==='iter' && res.details?.series){ Plots.series('plot', res.details.series, 'Residual norm per iteration'); }
    else if(WS==='interp' && res.details?.pts){ Plots.scatterFit('plot', res.details.pts, res.details.poly || null, res.details.spl || null); }
    else { Plots.blank('plot','See outputs at left.'); }

    History.add({time:Date.now(), ws:WS, method:id, fun:res.details?.fsrc||null, summary:res.summary});
    History.render();
    renderGuide();

  }catch(e){
    if(e instanceof Recommendation) statusBadge('warn', e.message);
    else statusBadge('err', e.message);
  }
}

const History={
  key:'nmh.history.v4',
  add(item){ const arr=this.get(); arr.unshift(item); localStorage.setItem(this.key, JSON.stringify(arr.slice(0,30))); },
  get(){ try{ return JSON.parse(localStorage.getItem(this.key)||'[]'); }catch{return [];} },
  clear(){ localStorage.removeItem(this.key); this.render(); },
  render(){
    const arr=this.get();
    els.history.innerHTML = arr.length? arr.map(h=>{
      const when=new Date(h.time).toLocaleString();
      return `<div class="flex items-center justify-between border-b py-2">
        <div>
          <div class="font-semibold">${when}</div>
          <div class="text-sm">${h.ws} • ${h.method}${h.fun? ' • f(x)='+h.fun:''}</div>
          <div class="text-sm text-slate-600">${h.summary}</div>
        </div>
      </div>`;
    }).join('') : '<div class="text-sm text-slate-600">No runs yet.</div>';
  }
};
document.getElementById('clearHistory').addEventListener('click', ()=>History.clear());

/* ============ Boot ============ */
function init(){ wireTabs(); setTab('root'); History.render(); }
if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded', init); } else { init(); }

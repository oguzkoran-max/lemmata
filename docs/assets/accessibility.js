(function(){
'use strict';
try {
var header=document.querySelector('.hd.intro-aware');
var intro=document.querySelector('.intro-sections');
if(header&&intro){
 function updateHeader(){header.classList.toggle('over-intro',intro.getBoundingClientRect().top<=header.offsetHeight);}
 window.addEventListener('scroll',updateHeader,{passive:true});
 window.addEventListener('resize',updateHeader,{passive:true});
 updateHeader();
}
var skipLink=document.querySelector('.hd .cine-summary');
if(skipLink){
 var skipTick=false;
 function skipState(){skipTick=false;var p=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--p'))||0;skipLink.style.pointerEvents=(p<0.18&&document.activeElement!==skipLink)?'none':'';}
 window.addEventListener('scroll',function(){if(!skipTick){skipTick=true;requestAnimationFrame(skipState);}},{passive:true});
 window.addEventListener('resize',function(){requestAnimationFrame(function(){requestAnimationFrame(skipState);});},{passive:true});
 skipLink.addEventListener('focus',skipState);skipLink.addEventListener('blur',skipState);skipState();
}
var tr=document.documentElement.lang==='tr';
var reduced=matchMedia('(prefers-reduced-motion: reduce)');
document.querySelectorAll('.motion-toggle').forEach(function(button){
 var v=document.getElementById(button.getAttribute('aria-controls'));
 var prev=button.previousElementSibling;if(prev&&prev.classList.contains('art'))prev.appendChild(button);
 function update(){var playing=!v.paused;var label=tr?(playing?'Animasyonu durdur':'Animasyonu oynat'):(playing?'Pause animation':'Play animation');button.setAttribute('aria-pressed',String(playing));button.setAttribute('aria-label',label);button.title=label;button.innerHTML=playing?'<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1" width="3" height="10" rx="1"/><rect x="7" y="1" width="3" height="10" rx="1"/></svg>':'<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9a.5.5 0 0 0 .77.42l7-4.5a.5.5 0 0 0 0-.84l-7-4.5A.5.5 0 0 0 3 1.5z"/></svg>';}
 if(reduced.matches){v.dataset.userPaused='true';v.pause();}
 button.addEventListener('click',function(){
  if(!v.paused){v.dataset.userPaused='true';v.pause();}
  else {v.dataset.userPaused='false';if(!v.getAttribute('src')&&v.dataset.src)v.src=v.dataset.src;v.classList.add('ready');v.play().catch(function(){});}
  update();
 });
 v.addEventListener('play',update);v.addEventListener('pause',update);update();
 reduced.addEventListener('change',function(e){if(e.matches){v.dataset.userPaused='true';v.pause();}});
});
/* a scroll-reveal block that receives keyboard focus stays shown, even if the observer never fired (B07) */
document.addEventListener('focusin',function(e){var r=e.target&&e.target.closest?e.target.closest('.reveal'):null;if(r)r.classList.add('in');});
var els=document.querySelectorAll('.intro-sections .reveal');
if(!('IntersectionObserver' in window)||reduced.matches)els.forEach(function(e){e.classList.add('in');});
else {var io=new IntersectionObserver(function(entries){entries.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}});});els.forEach(function(e){io.observe(e);});}
} catch(error) {
 document.querySelectorAll('.intro-sections .reveal').forEach(function(e){e.classList.add('in');e.style.opacity='1';e.style.transform='none';e.style.visibility='visible';});
}
})();

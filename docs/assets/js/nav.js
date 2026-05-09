/* Chapter navigation — prev/next keyboard shortcuts, active highlighting */
(function () {
  'use strict';

  const CHAPTERS = [
    { path: '/chapters/01-nature-and-math/index.html', title: 'Nature & Math' },
    { path: '/chapters/02-intuition/index.html',       title: 'Building Intuition' },
    { path: '/chapters/03-fourier-story/index.html',   title: "Fourier's Story" },
    { path: '/chapters/04-heat-equation/index.html',   title: 'The Heat Equation' },
    { path: '/chapters/05-pde-classification/index.html', title: 'PDE Classification' },
    { path: '/chapters/06-elliptic/index.html',        title: 'Elliptic PDEs' },
    { path: '/chapters/07-parabolic/index.html',       title: 'Parabolic PDEs' },
    { path: '/chapters/08-hyperbolic/index.html',      title: 'Hyperbolic PDEs' },
  ];

  function currentIndex() {
    const p = window.location.pathname;
    return CHAPTERS.findIndex(c => p.endsWith(c.path) || p.includes(c.path.replace('/index.html', '')));
  }

  function resolveRoot() {
    /* Works whether deployed at /nature-and-math/ or at / (local) */
    const p = window.location.pathname;
    const depth = (p.match(/\/chapters\/[^/]+\//)) ? '../../' : '';
    return depth;
  }

  function injectNavFooter(idx) {
    const footer = document.querySelector('.chapter-nav-footer');
    if (!footer) return;

    const root = resolveRoot();
    const prev = CHAPTERS[idx - 1];
    const next = CHAPTERS[idx + 1];

    if (prev) {
      footer.insertAdjacentHTML('afterbegin',
        `<a href="${root}${prev.path.replace(/^\//, '')}" class="prev">
           <span class="nav-label">&larr; Previous</span>
           <span class="nav-title">${prev.title}</span>
         </a>`);
    } else {
      footer.insertAdjacentHTML('afterbegin',
        `<a href="${root}index.html" class="prev">
           <span class="nav-label">&larr;</span>
           <span class="nav-title">Home</span>
         </a>`);
    }

    if (next) {
      footer.insertAdjacentHTML('beforeend',
        `<a href="${root}${next.path.replace(/^\//, '')}" class="next">
           <span class="nav-label">Next &rarr;</span>
           <span class="nav-title">${next.title}</span>
         </a>`);
    }
  }

  function setupKeyboard(idx) {
    const root = resolveRoot();
    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if ((e.key === 'ArrowLeft' || e.key === 'h') && !e.metaKey && !e.ctrlKey) {
        const prev = CHAPTERS[idx - 1];
        if (prev) window.location.href = root + prev.path.replace(/^\//, '');
        else window.location.href = root + 'index.html';
      }
      if ((e.key === 'ArrowRight' || e.key === 'l') && !e.metaKey && !e.ctrlKey) {
        const next = CHAPTERS[idx + 1];
        if (next) window.location.href = root + next.path.replace(/^\//, '');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    const idx = currentIndex();
    if (idx === -1) return; // not a chapter page
    injectNavFooter(idx);
    setupKeyboard(idx);
  });
})();

/* Pair picker: turns any <input data-pairs> (a comma-separated list like "SOLUSDT, AVAXUSDT") into
   removable chips plus a search box over every USDT perpetual Bitget lists (/api/pairs). The original
   input stays in the form as the value that is submitted, and is shown as-is if this script can't run. */
(function () {
  const fields = document.querySelectorAll('input[data-pairs]');
  if (!fields.length) return;
  const MAX = 12;
  let market = [];
  const norm = raw => {
    const s = raw.toUpperCase().replace(/[^A-Z0-9]/g, '');
    return !s || s === 'USDT' ? '' : (s.endsWith('USDT') ? s : s + 'USDT');
  };
  const short = sym => sym.replace(/USDT$/, '');
  const vol = v => v >= 1e9 ? '$' + (v / 1e9).toFixed(1) + 'B' : v >= 1e6 ? '$' + Math.round(v / 1e6) + 'M' : '$' + Math.round(v / 1e3) + 'K';

  const list = document.createElement('datalist');
  list.id = 'pair-options';
  document.body.append(list);
  const known = sym => market.some(m => m.symbol === sym);
  const ready = fetch('/api/pairs').then(r => r.json()).then(d => {
    market = d;
    d.forEach(m => { const o = document.createElement('option'); o.value = m.symbol; o.label = m.base + ' · 24h volume ' + vol(m.volume); list.append(o); });
  }).catch(() => {});

  fields.forEach(input => {
    let pairs = [...new Set(input.value.split(/[,\s]+/).map(norm).filter(Boolean))];
    const wrap = document.createElement('div'); wrap.className = 'pair-picker';
    const chips = document.createElement('div'); chips.className = 'chips';
    const row = document.createElement('div'); row.className = 'pair-add';
    const box = document.createElement('input'); box.type = 'text'; box.autocomplete = 'off';
    box.placeholder = 'Search a coin: ETH, DOGE, PEPE…'; box.setAttribute('list', 'pair-options');
    const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'ghost'; btn.textContent = 'Add';
    const msg = document.createElement('div'); msg.className = 'hint';
    const pop = document.createElement('div'); pop.className = 'chips popular';
    row.append(box, btn); wrap.append(chips, row, msg, pop);
    input.hidden = true; input.after(wrap);

    function render() {
      input.value = pairs.join(', ');
      input.dispatchEvent(new Event('change', { bubbles: true }));
      chips.replaceChildren(...pairs.map(p => {
        const c = document.createElement('span'); c.className = 'chip'; c.textContent = short(p);
        const x = document.createElement('button'); x.type = 'button'; x.textContent = '×'; x.title = 'Remove ' + p;
        x.addEventListener('click', () => { pairs = pairs.filter(q => q !== p); msg.textContent = ''; render(); });
        c.append(x); return c;
      }));
      if (!pairs.length) chips.textContent = 'No pairs chosen yet.';
      const top = market.filter(m => !pairs.includes(m.symbol)).slice(0, 8);
      pop.replaceChildren(...(top.length ? [Object.assign(document.createElement('span'), { className: 'muted', textContent: 'Popular: ' })] : []),
        ...top.map(m => {
          const b = document.createElement('button'); b.type = 'button'; b.className = 'chip add'; b.textContent = '+ ' + m.base;
          b.title = m.symbol + ' · 24h volume ' + vol(m.volume); b.addEventListener('click', () => tryAdd(m.symbol)); return b;
        }));
    }
    function tryAdd(raw) {
      const sym = norm(raw);
      if (!sym) return;
      msg.textContent = '';
      if (pairs.includes(sym)) { msg.textContent = short(sym) + ' is already added.'; }
      else if (market.length && !known(sym)) { msg.textContent = sym + " isn't a USDT perpetual on Bitget."; return; }
      else if (pairs.length >= MAX) { msg.textContent = 'At most ' + MAX + ' pairs per strategy.'; return; }
      else pairs.push(sym);
      box.value = ''; render();
    }
    btn.addEventListener('click', () => tryAdd(box.value));
    box.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); tryAdd(box.value); } });
    box.addEventListener('input', () => { if (known(box.value.toUpperCase())) tryAdd(box.value); });   // picked from the suggestions
    render();
    ready.then(render);
  });
})();

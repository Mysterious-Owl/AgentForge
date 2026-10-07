// Runs a page's own <script> in Node against a minimal fake DOM, calls its render functions
// with hostile strings, and prints every HTML string the page wrote (JSON on stdout).
// tests/test_pages.py drives it: an injected tag in that output is an XSS hole.
//     node tests/page_harness.js index.html fixtures.json
const fs = require('fs');
const vm = require('vm');

const [page, fixturesPath] = process.argv.slice(2);
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));
const html = fs.readFileSync(page, 'utf8');
const script = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));

const written = [];
// Every element is the same forgiving fake: any method is a no-op returning another fake,
// lists are empty, and every HTML write is recorded.
function fake() {
  const store = {};
  const fn = function () { return fake(); };
  return new Proxy(fn, {
    get(_, key) {
      if (key === Symbol.iterator) return function* () {};
      if (key === 'forEach') return () => {};
      if (key === 'length') return 0;
      if (key === 'innerHTML' || key === 'textContent' || key === 'value') return store[key] || '';
      if (key === 'insertAdjacentHTML') return (_pos, h) => written.push(String(h));
      if (key === 'then') return undefined;           // never mistaken for a promise
      if (key in store) return store[key];
      return fake();
    },
    set(_, key, value) {
      if (key === 'innerHTML' || key === 'outerHTML') written.push(String(value));
      store[key] = value;
      return true;
    },
    apply() { return fake(); },
  });
}

const noStore = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const sandbox = {
  document: fake(), window: { location: { origin: 'http://localhost:8000' }, open: () => {} },
  sessionStorage: noStore, localStorage: noStore, console,
  fetch: () => new Promise(() => {}),                 // nothing resolves: no network, no init race
  setTimeout: () => 0, clearTimeout: () => {}, crypto: {}, TextEncoder, TextDecoder,
  URL, atob, btoa,
};
vm.createContext(sandbox);
vm.runInContext(script + `
  ;globalThis.__render = {
    renderAnswer: typeof renderAnswer === 'function' ? renderAnswer : null,
    renderBuilds: typeof renderBuilds === 'function' ? renderBuilds : null,
    renderAction: typeof renderAction === 'function' ? renderAction : null,
    renderProfile: typeof renderProfile === 'function' ? renderProfile : null,
    addLog: typeof addLog === 'function' ? addLog : null,
    showError: typeof showError === 'function' ? showError : null,
    gateChips: typeof gateChips === 'function' ? gateChips : null,
    canonical: typeof canonical === 'function' ? canonical : null,
  };`, sandbox);

const before = written.length;
for (const [name, arg] of fixtures) {
  const fn = sandbox.__render[name];
  if (!fn) throw new Error(`${page} has no ${name}()`);
  // A function that RETURNS markup (gateChips) or text (canonical) is recorded like a write.
  const out = name === 'showError' ? fn(new Error(arg)) : name === 'addLog' ? fn('info', arg) : fn(arg);
  if (typeof out === 'string') written.push(out);
}
process.stdout.write(JSON.stringify(written.slice(before)));

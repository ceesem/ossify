// VertexExplorer front end: embedding scatter with lasso, linked to a 3D skeleton.
// Python owns the data; this file draws and turns gestures into `selected` (point
// indices). Each skeleton edge belongs to its child vertex, so a vertex's colour and
// selection paint the segment up to its parent.
//
// Labels (primary / terminal) and cell changes go to Python as messages; Python
// owns both and sends back `label_codes` and a whole new cell.

const PICK_PX = 7;
const GAP_HOPS = 10;             // "Fill gaps": the most unselected vertices a filled gap can hold
const W = 460, H = 460, PAD = 14;
const GREY = "#9a9890";
const LABEL_NAMES = ["primary", "terminal"];        // code k + 1, as in vertex.py LABELS
const LABEL_VARS = ["--vx-primary", "--vx-terminal"];

function h(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k !== null && k !== undefined) n.append(k);
  return n;
}

function fmt(v) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e4)) return v.toExponential(2);
  return v.toPrecision(3).replace(/\.?0+$/, "");
}

function cssVar(name, node, fallback = "#888") {
  return getComputedStyle(node).getPropertyValue(name).trim() || fallback;
}

function fitCanvas(canvas, w, hgt) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(hgt * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${hgt}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hgt);
  return ctx;
}

function inPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function render({ model, el }) {
  el.classList.add("cx-root", "vx-root");
  const S = {
    hover: -1, lasso: null, sel: new Set(),
    yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0, drag: null,
    mode: "hue", showContext: true, showSyn: true, dist: null,
  };

  // -- layout ----------------------------------------------------------------
  const scatter = h("canvas", { class: "cx-scatter" });
  const tip = h("div", { class: "cx-tip" });
  const morph = h("canvas", { class: "vx-morph" });
  const mtip = h("div", { class: "cx-tip" });
  const legend = h("div", { class: "vx-legend" });
  const status = h("div", { class: "cx-status" });
  const morphWrap = h("div", { class: "cx-scatter-wrap vx-morph-wrap" }, morph, mtip);

  const btn = (label, fn, title) => h("button", { class: "cx-btn", onclick: fn, title }, label);
  const chip = (label, on, fn) => {
    const b = h("button", { class: `cx-chip${on ? " on" : ""}`, "aria-pressed": String(on) }, label);
    b.addEventListener("click", () => {
      const now = !b.classList.contains("on");
      b.classList.toggle("on", now); b.setAttribute("aria-pressed", String(now)); fn(now);
    });
    return b;
  };
  const view = (yaw, pitch) => () => { Object.assign(S, { yaw, pitch, zoom: 1, panX: 0, panY: 0 }); projectMorph(); drawMorph(); };
  const contextChip = chip("context", true, on => { S.showContext = on; drawMorph(); });
  const synChip = chip("synapses", true, on => { S.showSyn = on; drawMorph(); });
  const modeSel = h("select", { class: "cx-select", title: "what colours the points and the skeleton" },
    ...[["hue", "colour: hue"], ["labels", "colour: labels"], ["plain", "colour: selection"]].map(([v, t]) =>
      h("option", { value: v }, t)));
  modeSel.addEventListener("change", () => { S.mode = modeSel.value; drawScatter(); drawMorph(); drawLegend(); });

  const send = msg => model.send(msg);
  const labelBtn = (k) => h("button", {
    class: "cx-btn vx-label-btn", title: `mark the selection ${LABEL_NAMES[k]} (overwrites)`,
    onclick: () => send({ action: "label", label: LABEL_NAMES[k] }),
  }, h("i", { style: `background:var(${LABEL_VARS[k]})` }), `Label ${LABEL_NAMES[k]}`);
  const navLabel = h("span", { class: "vx-nav-label" });
  const nav = h("span", { class: "vx-nav" },
    btn("◀", () => send({ action: "goto", step: -1 }), "previous cell"),
    navLabel,
    btn("▶", () => send({ action: "goto", step: 1 }), "next cell"));
  const note = h("span", { class: "cx-hint" });

  el.append(
    h("div", { class: "cx-toolbar vx-actions" },
      nav,
      btn("Longest path to root", longestPath,
        "keep only the path from the farthest tip of the selected piece nearest the root, back up to its top"),
      btn(`Fill gaps ≤ ${GAP_HOPS}`, fillGaps,
        `fill runs of up to ${GAP_HOPS} unselected vertices between selected ones; never grows past the selection`),
      btn("Fill gaps to root", fillToRoot,
        "add every vertex between the selection and the axon root, so each selected vertex is joined to it"),
      labelBtn(0), labelBtn(1),
      btn("Unlabel", () => send({ action: "label", label: null }), "remove labels from the selection"),
      note),
    h("div", { class: "cx-top" },
      h("div", { class: "cx-left" },
        h("div", { class: "cx-toolbar" },
          btn("Clear", () => setSelected([])),
          btn("Invert", () => {
            const n = model.get("points").x.length;
            const out = [];
            for (let i = 0; i < n; i++) if (!S.sel.has(i)) out.push(i);
            setSelected(out);
          }),
          h("span", { class: "cx-hint" }, "drag: lasso · shift: add · alt: remove")),
        h("div", { class: "cx-scatter-wrap" }, scatter, tip),
        legend),
      h("div", { class: "cx-right" },
        h("div", { class: "cx-toolbar" },
          btn("XY", view(0, 0), "x right, y down"),
          btn("ZY", view(Math.PI / 2, 0), "z right, y down"),
          btn("XZ", view(0, -Math.PI / 2), "from above"),
          modeSel,
          contextChip,
          synChip,
          h("span", { class: "cx-hint" }, "drag: rotate · shift-drag: pan · scroll: zoom · click: pick")),
        morphWrap,
        status),
    ),
  );

  // One colour rule for both panels; selection is shown by weight and fading.
  function colorOf(i, isSel) {
    if (i < 0) return GREY;
    if (S.mode === "labels") {
      const code = model.get("label_codes")[i] || 0;
      return code ? cssVar(LABEL_VARS[code - 1], el) : GREY;
    }
    if (S.mode === "plain") return isSel ? cssVar("--cx-accent", el) : GREY;
    const P = model.get("points");
    return P.palette[P.c[i]] || GREY;
  }

  // -- scatter -----------------------------------------------------------------
  let px = [], py = [];

  function projectScatter() {
    const P = model.get("points");
    const xs = P.x, ys = P.y;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]);
      y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]);
    }
    const k = Math.min((W - 2 * PAD) / (x1 - x0 || 1), (H - 2 * PAD) / (y1 - y0 || 1));
    const ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
    px = xs.map(v => ox + (v - x0) * k);
    py = ys.map(v => H - oy - (v - y0) * k);
  }

  function drawScatter() {
    const ctx = fitCanvas(scatter, W, H);
    const any = S.sel.size > 0;
    const n = px.length;
    const r = n > 5000 ? 1.6 : n > 1500 ? 2.2 : 3;
    const ink = cssVar("--cx-fg", el);
    // Unselected first, so the selection is never buried; batched by colour.
    for (const pass of [false, true]) {
      if (pass && !any) break;
      const byColor = new Map();
      for (let i = 0; i < n; i++) {
        if (any && S.sel.has(i) !== pass) continue;
        const c = colorOf(i, pass);
        if (!byColor.has(c)) byColor.set(c, []);
        byColor.get(c).push(i);
      }
      ctx.globalAlpha = any && !pass ? 0.15 : 0.85;
      for (const [c, idx] of byColor) {
        ctx.fillStyle = c;
        ctx.beginPath();
        for (const i of idx) { ctx.moveTo(px[i] + r, py[i]); ctx.arc(px[i], py[i], r, 0, 2 * Math.PI); }
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    if (S.hover >= 0) {
      ctx.beginPath(); ctx.arc(px[S.hover], py[S.hover], r + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.stroke();
    }
    if (S.lasso && S.lasso.length > 1) {
      ctx.beginPath();
      ctx.moveTo(...S.lasso[0]);
      for (const p of S.lasso.slice(1)) ctx.lineTo(...p);
      ctx.closePath();
      ctx.fillStyle = cssVar("--cx-wash", el); ctx.fill();
      ctx.setLineDash([4, 3]); ctx.strokeStyle = cssVar("--cx-accent", el); ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function nearestScatter(x, y) {
    let best = -1, bd = PICK_PX * PICK_PX;
    for (let i = 0; i < px.length; i++) {
      const d = (px[i] - x) ** 2 + (py[i] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  function setSelected(ids) {
    model.set("selected", ids);
    model.save_changes();
  }

  function combine(ids, e) {
    if (e.shiftKey) { const cur = new Set(S.sel); ids.forEach(i => cur.add(i)); return [...cur]; }
    if (e.altKey) { const drop = new Set(ids); return [...S.sel].filter(i => !drop.has(i)); }
    return ids;
  }

  function local(e, canvas) {
    const b = canvas.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top];
  }

  function setHover(i) {
    if (i === S.hover) return;
    S.hover = i;
    drawScatter(); drawMorph();
  }

  scatter.addEventListener("pointerdown", e => {
    scatter.setPointerCapture(e.pointerId);
    S.lasso = [local(e, scatter)];
  });
  scatter.addEventListener("pointermove", e => {
    const [x, y] = local(e, scatter);
    if (S.lasso) { S.lasso.push([x, y]); drawScatter(); return; }
    const i = nearestScatter(x, y);
    setHover(i);
    if (i >= 0) {
      tip.textContent = describe(i);
      tip.style.left = `${px[i] + 10}px`; tip.style.top = `${py[i] - 8}px`;
      tip.style.display = "block";
    } else tip.style.display = "none";
  });
  scatter.addEventListener("pointerup", e => {
    const path = S.lasso || [];
    S.lasso = null;
    let len = 0;
    for (let i = 1; i < path.length; i++) len += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    if (len < 6) {
      const i = nearestScatter(...local(e, scatter));
      if (i >= 0) setSelected(combine([i], e));
      else if (!e.shiftKey && !e.altKey) setSelected([]);
    } else {
      const ids = [];
      for (let i = 0; i < px.length; i++) if (inPolygon(px[i], py[i], path)) ids.push(i);
      setSelected(combine(ids, e));
    }
    drawScatter();
  });
  scatter.addEventListener("pointerleave", () => {
    if (!S.lasso) { tip.style.display = "none"; setHover(-1); }
  });

  function describe(i) {
    const P = model.get("points");
    return `vertex ${model.get("morph").vindex?.[P.v[i]] ?? P.v[i]} · ${P.names[0]} ${fmt(P.x[i])}, ${P.names[1]} ${fmt(P.y[i])}`;
  }

  function drawLegend() {
    const L = model.get("legend");
    legend.replaceChildren();
    if (S.mode === "labels") {
      const codes = model.get("label_codes");
      LABEL_NAMES.forEach((name, k) => {
        const ids = [];
        codes.forEach((c, i) => { if (c === k + 1) ids.push(i); });
        legend.append(h("span", { class: "cx-leg", title: "click: select these (shift adds)",
          onclick: e => setSelected(combine(ids, e)) },
          h("i", { style: `background:var(${LABEL_VARS[k]})` }), `${name} (${ids.length})`));
      });
      return;
    }
    if (S.mode === "plain") return;
    if (L.kind === "numeric") {
      legend.append(
        h("span", { class: "cx-hint" }, L.name),
        h("span", { class: "cx-num" }, fmt(L.lo)),
        h("i", { class: "vx-ramp", style: `background:linear-gradient(to right, ${L.ramp.join(",")})` }),
        h("span", { class: "cx-num" }, fmt(L.hi)));
    } else if (L.kind === "categorical") {
      legend.append(h("span", { class: "cx-hint" }, L.name));
      L.names.forEach((n, k) => legend.append(
        h("span", { class: "cx-leg", title: "click: select this label (shift adds)",
          onclick: e => {
            const P = model.get("points");
            const ids = [];
            for (let i = 0; i < P.c.length; i++) if (P.c[i] === k) ids.push(i);
            setSelected(combine(ids, e));
          } },
          h("i", { style: `background:${L.colors[k]}` }), n)));
    }
  }

  // -- morphology --------------------------------------------------------------
  // Orthographic: rotate the centred um coordinates by yaw (about y) then pitch
  // (about x); screen x = rotated x, screen y = rotated y (down, like the data).
  let MW = 520, MH = H;
  let fitK = 1;
  let mx = null, my = null, cx_ = null, cy_ = null;   // projected main / context vertices
  let sx = null, sy = null;                           // projected synapses

  function fitScale() {
    const M = model.get("morph");
    let r = 0;
    for (const arr of [M.xyz, M.context?.xyz || []]) {
      for (let i = 0; i < arr.length; i += 3) r = Math.max(r, Math.hypot(arr[i], arr[i + 1], arr[i + 2]));
    }
    fitK = (Math.min(MW, MH) / 2 - PAD) / (r || 1);
  }

  function projectArr(xyz, outX, outY) {
    const cy = Math.cos(S.yaw), sy = Math.sin(S.yaw), cp = Math.cos(S.pitch), sp = Math.sin(S.pitch);
    const k = fitK * S.zoom, ox = MW / 2 + S.panX, oy = MH / 2 + S.panY;
    for (let i = 0, j = 0; i < xyz.length; i += 3, j++) {
      const x = xyz[i], y = xyz[i + 1], z = xyz[i + 2];
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      const y2 = cp * y - sp * z1;
      outX[j] = ox + k * x1;
      outY[j] = oy + k * y2;
    }
  }

  function projectMorph() {
    const M = model.get("morph");
    const n = M.xyz.length / 3;
    if (!mx || mx.length !== n) { mx = new Float32Array(n); my = new Float32Array(n); }
    projectArr(M.xyz, mx, my);
    if (M.context) {
      const m = M.context.xyz.length / 3;
      if (!cx_ || cx_.length !== m) { cx_ = new Float32Array(m); cy_ = new Float32Array(m); }
      projectArr(M.context.xyz, cx_, cy_);
    }
    if (M.synapses) {
      const m = M.synapses.xyz.length / 3;
      if (!sx || sx.length !== m) { sx = new Float32Array(m); sy = new Float32Array(m); }
      projectArr(M.synapses.xyz, sx, sy);
    }
  }

  function strokeEdges(ctx, X, Y, parent, idx) {
    ctx.beginPath();
    for (const v of idx) {
      const p = parent[v];
      if (p < 0) continue;
      ctx.moveTo(X[v], Y[v]); ctx.lineTo(X[p], Y[p]);
    }
    ctx.stroke();
  }

  function drawMorph() {
    const M = model.get("morph");
    const P = model.get("points");
    if (!mx) return;
    const ctx = fitCanvas(morph, MW, MH);
    ctx.lineCap = "round";
    const lw = Math.max(0.6, Math.min(2.2, 0.9 * Math.sqrt(S.zoom)));

    if (M.context && S.showContext) {
      ctx.strokeStyle = cssVar("--cx-muted", el);
      ctx.globalAlpha = 0.35; ctx.lineWidth = lw * 0.8;
      const all = [];
      for (let i = 0; i < cx_.length; i++) all.push(i);
      strokeEdges(ctx, cx_, cy_, M.context.parent, all);
      ctx.globalAlpha = 1;
      if (M.context.root >= 0) {
        ctx.fillStyle = cssVar("--cx-fg", el);
        ctx.beginPath(); ctx.arc(cx_[M.context.root], cy_[M.context.root], 4, 0, 2 * Math.PI); ctx.fill();
      }
    }

    // Group vertices by (selected?, colour) so each group is one stroke.
    const any = S.sel.size > 0;
    const groups = new Map();
    const n = mx.length;
    for (let v = 0; v < n; v++) {
      const i = M.point[v];
      const isSel = i >= 0 && S.sel.has(i);
      const color = colorOf(i, isSel);
      const key = `${isSel ? 1 : 0}|${color}`;
      if (!groups.has(key)) groups.set(key, { isSel, color, idx: [] });
      groups.get(key).idx.push(v);
    }
    for (const pass of [false, true]) {
      for (const g of groups.values()) {
        if (g.isSel !== pass) continue;
        ctx.strokeStyle = g.color;
        ctx.globalAlpha = any && !g.isSel ? 0.18 : 1;
        ctx.lineWidth = g.isSel ? lw * 2.4 : lw;
        strokeEdges(ctx, mx, my, M.parent, g.idx);
        if (g.isSel) {
          // Dots too: a lone selected vertex has no selected edge to show it.
          ctx.fillStyle = g.color;
          ctx.beginPath();
          const r = lw * 1.4;
          for (const v of g.idx) { ctx.moveTo(mx[v] + r, my[v]); ctx.arc(mx[v], my[v], r, 0, 2 * Math.PI); }
          ctx.fill();
        }
      }
    }
    ctx.globalAlpha = 1;

    // Synapses at their own positions; with a selection, those on selected vertices
    // stay bright. One fill per alpha level.
    if (M.synapses && S.showSyn) {
      const r = Math.max(1.6, Math.min(3.2, 1.8 * Math.sqrt(S.zoom)));
      ctx.fillStyle = cssVar("--vx-syn", el, "#d62728");
      for (const bright of [false, true]) {
        ctx.globalAlpha = bright ? 0.95 : (any ? 0.2 : 0.95);
        ctx.beginPath();
        for (let k = 0; k < sx.length; k++) {
          const v = M.synapses.vertex[k];
          const on = !any || (v >= 0 && S.sel.has(M.point[v]));
          if (on !== bright) continue;
          ctx.moveTo(sx[k] + r, sy[k]); ctx.arc(sx[k], sy[k], r, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    if (!M.context && M.root >= 0) {
      ctx.strokeStyle = cssVar("--cx-fg", el); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(mx[M.root], my[M.root], 4, 0, 2 * Math.PI); ctx.stroke();
    }
    if (S.hover >= 0) {
      const v = P.v[S.hover];
      ctx.strokeStyle = cssVar("--cx-fg", el); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(mx[v], my[v], 7, 0, 2 * Math.PI); ctx.stroke();
    }

    // Scale bar: the largest round length under a quarter of the width.
    const k = fitK * S.zoom;
    const target = (MW / 4) / k;
    const pow = 10 ** Math.floor(Math.log10(target));
    const um = [5, 2, 1].map(m => m * pow).find(u => u <= target) || pow;
    const y = MH - 10, x1 = MW - 10, x0 = x1 - um * k;
    ctx.strokeStyle = cssVar("--cx-fg", el); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillStyle = cssVar("--cx-fg", el); ctx.font = "10px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`${um} µm`, x1, y - 4);
  }

  function nearestVertex(x, y) {
    const M = model.get("morph");
    let best = -1, bd = PICK_PX * PICK_PX;
    for (let v = 0; v < mx.length; v++) {
      if (M.point[v] < 0) continue;
      const d = (mx[v] - x) ** 2 + (my[v] - y) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  }

  morph.addEventListener("pointerdown", e => {
    morph.setPointerCapture(e.pointerId);
    const [x, y] = local(e, morph);
    S.drag = { x, y, x0: x, y0: y, pan: e.shiftKey, moved: false };
    mtip.style.display = "none";
  });
  morph.addEventListener("pointermove", e => {
    const [x, y] = local(e, morph);
    if (S.drag) {
      const dx = x - S.drag.x, dy = y - S.drag.y;
      S.drag.x = x; S.drag.y = y;
      if (Math.hypot(x - S.drag.x0, y - S.drag.y0) > 3) S.drag.moved = true;
      if (S.drag.pan) { S.panX += dx; S.panY += dy; }
      else {
        S.yaw += dx * 0.01;
        S.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, S.pitch - dy * 0.01));
      }
      projectMorph(); drawMorph();
      return;
    }
    const v = nearestVertex(x, y);
    const i = v >= 0 ? model.get("morph").point[v] : -1;
    setHover(i);
    if (i >= 0) {
      mtip.textContent = describe(i);
      mtip.style.left = `${x + 12}px`; mtip.style.top = `${y + 12}px`;
      mtip.style.display = "block";
    } else mtip.style.display = "none";
  });
  morph.addEventListener("pointerup", e => {
    const d = S.drag;
    S.drag = null;
    if (d && !d.moved) {
      const v = nearestVertex(...local(e, morph));
      const i = v >= 0 ? model.get("morph").point[v] : -1;
      if (i >= 0) setSelected(combine([i], e));
    }
  });
  morph.addEventListener("pointerleave", () => {
    if (!S.drag) { mtip.style.display = "none"; setHover(-1); }
  });
  morph.addEventListener("dblclick", view(0, 0));
  morph.addEventListener("wheel", e => {
    e.preventDefault();
    const [x, y] = local(e, morph);
    const f = Math.exp(-e.deltaY * 0.0015);
    // Zoom about the cursor: keep the point under it fixed.
    const cx = MW / 2 + S.panX, cy = MH / 2 + S.panY;
    S.panX = x + (cx - x) * f - MW / 2;
    S.panY = y + (cy - y) * f - MH / 2;
    S.zoom *= f;
    projectMorph(); drawMorph();
  }, { passive: false });

  // -- longest path to root ------------------------------------------------------------
  // Path length from the skeleton root to every vertex, along the tree (um).
  function distances(M) {
    if (S.dist && S.dist.M === M) return S.dist.d;
    const n = M.parent.length, d = new Float64Array(n).fill(-1), X = M.xyz;
    for (let v = 0; v < n; v++) {
      const path = [];
      for (let u = v; u >= 0 && d[u] < 0; u = M.parent[u]) path.push(u);
      for (let k = path.length - 1; k >= 0; k--) {
        const w = path[k], p = M.parent[w];
        d[w] = p < 0 ? 0 : d[p] + Math.hypot(X[3 * w] - X[3 * p], X[3 * w + 1] - X[3 * p + 1], X[3 * w + 2] - X[3 * p + 2]);
      }
    }
    S.dist = { M, d };
    return d;
  }

  // Within the selection, take the connected piece containing the axon root (or, if
  // the root is not selected, the piece whose top is nearest it), find its farthest
  // vertex from the root, and keep only the path from there up to the piece's top.
  function longestPath() {
    const M = model.get("morph");
    if (!S.sel.size) { note.textContent = "select something first"; return; }
    const n = M.parent.length;
    const on = new Uint8Array(n);
    for (let v = 0; v < n; v++) if (M.point[v] >= 0 && S.sel.has(M.point[v])) on[v] = 1;
    const d = distances(M);
    const top = new Int32Array(n).fill(-2);
    const topOf = v => {
      const path = [];
      let u = v;
      while (top[u] === -2) {
        path.push(u);
        const p = M.parent[u];
        if (p < 0 || !on[p]) { top[u] = u; break; }
        u = p;
      }
      const t = top[u];
      for (const w of path) top[w] = t;
      return t;
    };
    let rootTop = -1;
    for (let v = 0; v < n; v++) {
      if (!on[v]) continue;
      const t = topOf(v);
      if (rootTop < 0 || d[t] < d[rootTop]) rootTop = t;
    }
    let tip = rootTop;
    for (let v = 0; v < n; v++) if (on[v] && top[v] === rootTop && d[v] > d[tip]) tip = v;
    const keep = [];
    for (let u = tip; ; u = M.parent[u]) {
      keep.push(M.point[u]);
      if (u === rootTop) break;
    }
    note.textContent = `path: ${keep.length} vertices, ${fmt(d[tip] - d[rootTop])} µm` +
      (M.parent[rootTop] < 0 ? " from the axon root" : ` starting ${fmt(d[rootTop])} µm past the axon root`);
    setSelected(keep);
  }

  // Fill runs of at most GAP_HOPS unselected vertices between selected ones. Every tree path
  // goes up to a lowest common ancestor and back down, so walk up from each selected
  // vertex: reaching a selected ancestor in time fills the run between; reaching an
  // unselected ancestor from two different children within the budget fills both runs
  // (a gap that straddles a branch point).
  function fillGaps() {
    const M = model.get("morph");
    if (!S.sel.size) { note.textContent = "select something first"; return; }
    const n = M.parent.length;
    const on = new Uint8Array(n);
    for (let v = 0; v < n; v++) if (M.point[v] >= 0 && S.sel.has(M.point[v])) on[v] = 1;
    const keep = new Set(S.sel);
    const fill = (from, to) => {           // from up to (excluding) `to`
      for (let u = from; u !== to && u >= 0; u = M.parent[u]) if (M.point[u] >= 0) keep.add(M.point[u]);
    };
    const reach = new Map();               // unselected ancestor -> Map(child -> {hops, start})
    for (let v = 0; v < n; v++) {
      if (!on[v]) continue;
      let child = v, hops = 0;
      for (let u = M.parent[v]; u >= 0 && hops <= GAP_HOPS; child = u, u = M.parent[u]) {
        hops++;
        if (on[u]) { if (hops > 1) fill(M.parent[v], u); break; }
        if (!reach.has(u)) reach.set(u, new Map());
        const byChild = reach.get(u), best = byChild.get(child);
        if (!best || hops < best.hops) byChild.set(child, { hops, start: v });
      }
    }
    for (const [w, byChild] of reach) {
      if (byChild.size < 2) continue;
      const two = [...byChild.values()].sort((a, b) => a.hops - b.hops).slice(0, 2);
      if (two[0].hops + two[1].hops - 1 > GAP_HOPS) continue;   // both runs plus w
      for (const { start } of two) fill(M.parent[start], M.parent[w]);
    }
    note.textContent = `filled ${keep.size - S.sel.size} vertices`;
    setSelected([...keep]);
  }

  // The converse of longestPath: add each selected vertex's whole path to the root.
  function fillToRoot() {
    const M = model.get("morph");
    if (!S.sel.size) { note.textContent = "select something first"; return; }
    const n = M.parent.length;
    const seen = new Uint8Array(n);
    const keep = new Set(S.sel);
    for (let v = 0; v < n; v++) {
      if (!(M.point[v] >= 0 && S.sel.has(M.point[v]))) continue;
      for (let u = v; u >= 0 && !seen[u]; u = M.parent[u]) {
        seen[u] = 1;
        if (M.point[u] >= 0) keep.add(M.point[u]);
      }
    }
    note.textContent = `filled ${keep.size - S.sel.size} vertices`;
    setSelected([...keep]);
  }

  // -- wiring -------------------------------------------------------------------------
  function syncSel() { S.sel = new Set(model.get("selected")); drawScatter(); drawMorph(); }
  function all() {
    S.sel = new Set(model.get("selected"));
    S.hover = -1;
    contextChip.style.display = model.get("morph").context ? "" : "none";
    synChip.style.display = model.get("morph").synapses ? "" : "none";
    projectScatter(); fitScale(); projectMorph();
    drawScatter(); drawMorph(); drawLegend();
    status.textContent = model.get("status");
  }
  model.on("change:points", all);
  model.on("change:morph", () => {
    Object.assign(S, { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 });
    note.textContent = "";
    all();
  });
  model.on("change:label_codes", () => { drawScatter(); drawMorph(); drawLegend(); });
  model.on("change:nav", drawNav);
  function drawNav() {
    const N = model.get("nav");
    nav.style.display = N && N.n ? "" : "none";
    if (N && N.n) navLabel.textContent = `cell ${N.index + 1} / ${N.n} · ${N.root_id}`;
  }
  drawNav();
  model.on("change:legend", drawLegend);
  model.on("change:selected", syncSel);
  model.on("change:status", () => { status.textContent = model.get("status"); });

  function resize() {
    const w = Math.max(320, Math.floor(morphWrap.clientWidth || 520));
    if (w === MW) return;
    MW = w; fitScale(); projectMorph(); drawMorph();
  }
  all();
  requestAnimationFrame(resize);
  const ro = new ResizeObserver(resize);
  ro.observe(morphWrap);
  return () => ro.disconnect();
}

export default { render };

// transfer fc: browse premier league player-seasons from data/players.json
// (built by scripts/export_site_data.py)

const state = { data: null, season: null, group: "all", sort: "value", query: "", list: [], current: null };

// lowercase and strip accents so "odegaard" finds "ødegaard" and "nunez" finds "núñez"
const fold = (s) => (s ?? "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
  .replace(/ø/g, "o").replace(/æ/g, "ae").replace(/ß/g, "ss").replace(/ł/g, "l").replace(/đ/g, "d");
const $ = (id) => document.getElementById(id);

const UCL = ["—", "league phase", "playoff", "last 16", "quarter-final", "semi-final", "final"];
const LEAGUES = {
  GB1: "premier league", ES1: "la liga", L1: "bundesliga", IT1: "serie a", FR1: "ligue 1",
  NL1: "eredivisie", PO1: "liga portugal", BE1: "belgian pro league", SC1: "scottish prem",
  TR1: "süper lig", BRA1: "brasileirão", MLS1: "mls", DK1: "superliga", A1: "austrian bundesliga",
};

function money(v) {
  if (v == null) return "—";
  if (v >= 1e6) {
    const m = v / 1e6;
    return `€${m >= 10 ? Math.round(m) : m.toFixed(1).replace(/\.0$/, "")}m`;
  }
  return `€${Math.round(v / 1e3)}k`;
}

const GROUP_NAMES = { fwd: "forwards", mid: "midfielders", def: "defenders", gk: "goalkeepers" };
const rankLabel = (p) => (p.score_rank == null ? "—" : `${p.score_rank}/${p.group_size} ${GROUP_NAMES[p.group]}`);

const seasonLabel = (s) => `${String(s).slice(2)}/${String(s + 1).slice(2)}`;

// --- theme -------------------------------------------------------------
function loadTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
}

// --- grid --------------------------------------------------------------
// tiles show the prediction; 22/23 has none (no earlier season to train on), so the
// actual value, which is the season-start value
const shownValue = (p) => p.predicted ?? p.values[2];
// forecast change from the season-start value (null without a prediction)
const change = (p) => (p.predicted != null && p.values[2] ? p.predicted / p.values[2] - 1 : null);
const FLAT = 0.03;  // changes smaller than this read as flat
const FEW_MINUTES = 270;  // under three full games: the forecast leans on very little this season
const lowConfidence = (p) => p.predicted != null && p.minutes < FEW_MINUTES;
const trend = (c) => (c == null ? "" : c > FLAT ? "up" : c < -FLAT ? "down" : "flat");
const arrow = (c) => (c > FLAT ? "↑" : c < -FLAT ? "↓" : "→");
const euroChange = (p) => (change(p) == null ? null : p.predicted - p.values[2]);
// risers/fallers show the euro move too, since that's what they're sorted by
const changeLabel = (p) => {
  const c = change(p);
  if (c == null) return "";
  const pct = `${Math.abs(Math.round(c * 100))}%`;
  return state.sort === "risers" || state.sort === "fallers"
    ? `${arrow(c)} ${money(Math.abs(euroChange(p)))} · ${pct}` : `${arrow(c)} ${pct}`;
};

function visiblePlayers() {
  let rows = state.data.players.filter((p) => p.season === state.season);
  if (state.group !== "all") rows = rows.filter((p) => p.group === state.group);
  if (state.query) rows = rows.filter((p) => fold(`${p.name} ${p.club}`).includes(state.query));
  const keys = {
    value: (p) => shownValue(p) ?? -1,
    // "all" compares positions by percentile; within one position the raw score works
    score: (p) => (state.group === "all" ? p.score_pct : p.score) ?? -1,
    // by euro change, so a €500k youngster's +600% doesn't top the list; players
    // without a forecast sink to the bottom of both
    risers: (p) => euroChange(p) ?? -Infinity,
    fallers: (p) => (euroChange(p) == null ? -Infinity : -euroChange(p)),
  };
  const key = keys[state.sort];
  return rows.sort((a, b) => key(b) - key(a));
}

// risers/fallers need forecasts; a season without any (22/23) greys them out
function updateSorts() {
  const hasForecasts = state.data.players.some((p) => p.season === state.season && p.predicted != null);
  $("sorts").querySelectorAll('[data-sort="risers"], [data-sort="fallers"]').forEach((b) => {
    b.disabled = !hasForecasts;
    b.title = hasForecasts ? "" : "no forecasts for this season";
  });
  if (!hasForecasts && (state.sort === "risers" || state.sort === "fallers")) {
    state.sort = "value";
    $("sorts").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.sort === "value"));
  }
}

function renderGrid() {
  updateSorts();
  const rows = visiblePlayers();
  const gw = state.data.live && state.season === state.data.live.season ? ` · after gw ${state.data.live.gameweek}` : "";
  $("count").textContent = `${rows.length} ${rows.length === 1 ? "player" : "players"}${gw}`;
  $("grid").innerHTML = rows.length || !state.query ? "" : `<p class="empty">no players found</p>`;
  state.list = rows;  // panel arrows step through what's on screen
  rows.forEach((p, i) => {
    const tile = document.createElement("button");
    tile.className = "tile";
    const c = change(p);
    if (c != null) tile.dataset.trend = trend(c);
    tile.style.animationDelay = `${Math.min(i, 24) * 18}ms`;
    tile.innerHTML = `
      <span class="rank">${String(i + 1).padStart(2, "0")}</span>
      <span class="art"><span class="initials">${p.initials}</span></span>
      <span class="value ${p.predicted == null ? "actual" : ""}">${state.sort === "score" ? rankLabel(p) : money(shownValue(p))}</span>
      <span class="name">${p.name}</span>
      <span class="club">${p.club ?? ""}</span>
      ${c != null && state.sort !== "score" ? `<span class="change ${trend(c)}${lowConfidence(p) ? " faint" : ""}"
        ${lowConfidence(p) ? `title="low confidence: few minutes"` : ""}>${changeLabel(p)}</span>` : ""}`;
    tile.addEventListener("click", () => openPanel(p));
    $("grid").appendChild(tile);
  });
}

function renderSeasons() {
  $("seasons").innerHTML = "";
  state.data.seasons.forEach((s) => {
    const b = document.createElement("button");
    b.textContent = seasonLabel(s);
    b.classList.toggle("on", s === state.season);
    b.addEventListener("click", () => { state.season = s; renderSeasons(); renderGrid(); });
    $("seasons").appendChild(b);
  });
}

function bindToggleGroup(containerId, attr, key) {
  $(containerId).querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      state[key] = b.dataset[attr];
      $(containerId).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      renderGrid();
    });
  });
}

// --- detail panel ------------------------------------------------------
const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const shortDate = (iso) => `${+iso.slice(8)} ${MONTHS[+iso.slice(5, 7) - 1]}`;

function trendSvg(p) {
  // big labelled dots: june 30 snapshots the model uses (3y ago, 2y ago, season start, season end);
  // small dots: every transfermarkt valuation in between, placed on a real time axis
  const years = [p.season - 2, p.season - 1, p.season, p.season + 1];
  const day = (iso) => Date.parse(iso) / 864e5;
  const t0 = day(`${years[0]}-06-30`), t1 = day(`${years[3]}-06-30`);
  const snaps = p.values.map((v, i) => ({ v, t: day(`${years[i]}-06-30`), label: `jun ${String(years[i]).slice(2)}` }));
  const known = snaps.filter((d) => d.v != null);
  // forecast: dashed line from the season-start value to the predicted season-end value
  const forecast = p.predicted != null && p.values[2] != null ? { v: p.predicted, t: snaps[3].t } : null;
  if (known.length + (forecast ? 1 : 0) < 2) return `<div class="note">not enough history</div>`;
  const updates = (state.data.valuations?.[p.id] ?? [])
    .map(([date, v]) => ({ v, t: day(date), date }))
    .filter((d) => d.t > Math.max(t0, known[0].t) && d.t < t1);
  const all = [...known, ...updates].sort((a, b) => a.t - b.t);
  const scale = forecast ? [...all, forecast] : all;
  const max = Math.max(...scale.map((d) => d.v)), min = Math.min(...scale.map((d) => d.v));
  const x = (t) => 30 + ((t - t0) / (t1 - t0)) * 318;
  const y = (v) => 90 - (max === min ? 40 : ((v - min) / (max - min)) * 70);
  const line = all.map((d) => `${x(d.t)},${y(d.v)}`).join(" ");
  // with an actual season-end value at the same spot, the higher of the two labels goes
  // above its dot and the lower one below, so they never overlap
  const endSnap = snaps[3].v != null ? snaps[3] : null;
  const forecastLabelY = !forecast ? 0 : !endSnap ? y(forecast.v) + 3.5
    : forecast.v > endSnap.v ? y(forecast.v) - 8 : y(forecast.v) + 14;
  return `<svg viewBox="0 0 380 120" preserveAspectRatio="none">
    <rect class="season" x="${x(snaps[2].t)}" y="0" width="${x(snaps[3].t) - x(snaps[2].t)}" height="100"/>
    <polyline class="line" points="${line}"/>
    ${updates.map((d) => `<circle class="dot small" cx="${x(d.t)}" cy="${y(d.v)}" r="1.8"><title>${shortDate(d.date)} ${d.date.slice(0, 4)} · ${money(d.v)}</title></circle>`).join("")}
    ${known.map((d) => `<circle class="dot" cx="${x(d.t)}" cy="${y(d.v)}" r="3"/>
      <text x="${x(d.t)}" y="${d === endSnap && forecast && forecast.v > d.v ? y(d.v) + 16 : y(d.v) - 10}" text-anchor="middle">${money(d.v)}</text>`).join("")}
    ${forecast ? `<line class="forecast" x1="${x(snaps[2].t)}" y1="${y(p.values[2])}" x2="${x(forecast.t)}" y2="${y(forecast.v)}"/>
      <circle class="dot forecast-dot" cx="${x(forecast.t)}" cy="${y(forecast.v)}" r="3"><title>forecast ${money(forecast.v)}</title></circle>
      <text class="forecast-label" x="${x(forecast.t) - 7}" y="${forecastLabelY}" text-anchor="end">${money(forecast.v)}</text>` : ""}
    ${snaps.map((d) => `<text x="${x(d.t)}" y="116" text-anchor="middle">${d.label}</text>`).join("")}
  </svg>
  ${forecast ? `<div class="legend"><span class="key solid"></span>transfermarkt <span class="key dashed"></span>forecast</div>` : ""}`;
}

// the season's own valuation updates, spelled out under the chart
// the season in progress (from the fantasy premier league feed), if this is it
const liveSeason = (p) => state.data.live && p.season === state.data.live.season;

function seasonUpdates(p) {
  if (liveSeason(p)) return `<div class="note">no valuations yet this season (source paused in june 2026)</div>`;
  const from = `${p.season}-06-30`, to = `${p.season + 1}-06-30`;
  const rows = (state.data.valuations?.[p.id] ?? []).filter(([date]) => date > from && date <= to);
  return rows.length
    ? `<div class="note">valuations during ${seasonLabel(p.season)}: ${rows.map(([d, v]) => `${shortDate(d)} ${money(v)}`).join(" · ")}</div>`
    : `<div class="note">no valuation updates during ${seasonLabel(p.season)}</div>`;
}

function predictionNote(p) {
  if (p.predicted == null) {
    return p.season === state.data.seasons[0] ? "no earlier season to learn from" : "no starting value to predict from";
  }
  const change = Math.round((p.predicted / p.values[2] - 1) * 100);
  const years = p.trained_on.split("-").map((y) => String(y).slice(2)).join("–");
  const basis = liveSeason(p) ? `forecast after gw ${state.data.live.gameweek}` : `learned from ${years}`;
  return `${change >= 0 ? "+" : "−"}${Math.abs(change)}% from start · ${basis}`;
}

// actual = market value at season start: the latest valuation on/before 30 jun before
// the season (the model's starting point, so no "% off" accuracy is shown against it)
function actualNote(p) {
  const cutoff = `${p.season}-06-30`;
  const last = (state.data.valuations?.[p.id] ?? []).filter(([date]) => date <= cutoff).at(-1);
  return last ? `season start · valued ${shortDate(last[0])} ${last[0].slice(2, 4)}` : "season start";
}

function factorsHtml(p) {
  const m = state.data.model;
  if (!m) return `<div class="note">run train_model.py</div>`;
  const r = m.results;
  const overall = `<div class="note">overall the model leans most on ${m.factors.slice(0, 3).map((f) => f.name).join(", ")}.
    on ${seasonLabel(m.test_season)} it's off by ~${Math.round((Math.exp(r.model.male) - 1) * 100)}% typically,
    vs ~${Math.round((Math.exp(r.baseline.male) - 1) * 100)}% for "value stays the same".</div>`;
  if (!p.factors.length) return `<div class="note">no prediction for this player-season.</div>${overall}`;

  // this player's reasons: each bar grows left (held back) or right (pushed up) from the centre
  const max = Math.max(...p.factors.map((f) => Math.abs(f.effect)));
  const rows = p.factors.map((f) => {
    const width = (Math.abs(f.effect) / max) * 50;
    const side = f.effect > 0 ? `left:50%` : `right:50%`;
    const pct = `${f.effect > 0 ? "+" : "−"}${Math.round(Math.abs(f.effect) * 100)}%`;
    return `<span>${f.name}</span><span class="fbar signed"><i style="${side};width:${width}%"></i></span><span class="pct">${pct}</span>`;
  }).join("");
  return `<div class="stats factors">${rows}</div>${overall}`;
}

function openPanel(p) {
  const prev = p.prev.minutes == null
    ? "no data"
    : `${p.prev.minutes} min · ${p.prev.goals}g ${p.prev.assists}a · ${LEAGUES[p.prev.league] ?? p.prev.league ?? ""}`;
  const stale = p.value_end_age_days > 180 ? `<div class="note">season-end value is ${p.value_end_age_days} days old — excluded from training</div>` : "";
  const stats = [
    ["age", p.age], ["apps", p.apps], ["minutes", p.minutes], ["goals", p.goals], ["assists", p.assists],
    ...(p.group === "def" || p.group === "gk" ? [["clean sheets", p.clean_sheets]] : []),
    ["cards", `${p.yellows}y ${p.reds}r`], [liveSeason(p) ? "club position so far" : "club finish", p.league_position ?? "—"],
    ["champions league", UCL[p.ucl_stage]], ["promoted club", p.promoted ? "yes" : "no"],
    ...(p.other_league_minutes ? [["other league min", p.other_league_minutes]] : []),
    ["last season", prev],
  ];
  $("panel-body").innerHTML = `
    <div class="sub">${seasonLabel(p.season)} · ${p.initials} · ${p.club ?? ""}</div>
    <h2>${p.name}</h2>
    <section class="vs">
      <div><div class="label">predicted${p.explanation ? ` <button class="why" id="why" aria-expanded="false">why ↓</button>` : ""}</div>
        <div class="big ${p.predicted == null ? "pending" : ""}" ${p.predicted != null ? `data-value="${p.predicted}"` : ""}>${money(p.predicted)}</div>
        <div class="note">${predictionNote(p)}</div>
        ${lowConfidence(p) ? `<div class="note caution">low confidence: only ${p.minutes} minutes${liveSeason(p) ? " so far" : ""}</div>` : ""}</div>
      <div><div class="label">actual</div><div class="big" ${p.values[2] != null ? `data-value="${p.values[2]}"` : ""}>${money(p.values[2])}</div>
        <div class="note">${actualNote(p)}</div>${stale}</div>
    </section>
    ${p.explanation ? `<div class="explain" id="explain" hidden>${p.why_ai
      ? `<p>${p.why_ai}</p><p class="details"><span class="tag">details</span> ${p.explanation}</p>`
      : `<p>${p.explanation}</p>`}</div>` : ""}
    <section>
      <div class="label">performance score</div>
      <div class="vs"><div class="big">${p.score ?? "—"}</div><div class="big">${p.score_rank ?? "—"}<span class="sub">/${p.group_size}</span></div></div>
      <div class="note">rank among ${p.group_size} ${GROUP_NAMES[p.group]} in ${seasonLabel(p.season)}</div>
    </section>
    <section class="trend"><div class="label">value trend</div>${trendSvg(p)}${seasonUpdates(p)}</section>
    <section><div class="label">season</div>
      <div class="stats">${stats.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>
    </section>
    <section><div class="label">factors</div>${factorsHtml(p)}</section>`;
  const why = $("why");
  if (why) {
    // the predicted value (and its "why" button) toggles the explanation
    const toggle = () => {
      const open = $("explain").hidden;
      $("explain").hidden = !open;
      why.setAttribute("aria-expanded", String(open));
      why.textContent = open ? "why ↑" : "why ↓";
    };
    const value = why.closest(".label").parentElement.querySelector(".big");
    why.addEventListener("click", toggle);
    value.addEventListener("click", toggle);
    value.classList.add("clickable");
  }
  $("panel-body").querySelectorAll(".vs .big[data-value]").forEach(countUp);
  state.current = p;
  setHash(playerHash(p));
  updateSteps();
  showPanel();
}

function showPanel() {
  const wasOpen = $("panel").classList.contains("open");
  $("panel").classList.add("open");
  $("panel").setAttribute("aria-hidden", "false");
  $("scrim").hidden = false;
  if (!wasOpen) $("panel-close").focus();
  $("panel").scrollTop = 0;
}

// previous/next player in the grid's current order (also ← → keys)
function updateSteps() {
  const i = state.current ? state.list.indexOf(state.current) : -1;
  $("panel-prev").disabled = i <= 0;
  $("panel-next").disabled = i < 0 || i >= state.list.length - 1;
}

function step(dir) {
  const i = state.list.indexOf(state.current);
  if (i < 0) return;
  const next = state.list[i + dir];
  if (next) openPanel(next);
}

// money values count up from zero when the panel opens (skipped for reduced motion)
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function countUp(el) {
  const target = Number(el.dataset.value);
  if (!target || reduceMotion()) return;
  const start = performance.now(), duration = 520;
  const frame = (now) => {
    const t = Math.min((now - start) / duration, 1);
    el.textContent = money(target * (1 - Math.pow(1 - t, 3)));  // ease-out
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// --- about -------------------------------------------------------------
function openAbout() {
  const m = state.data.model, live = state.data.live;
  const pct = (male) => Math.round((Math.exp(male) - 1) * 100);
  const accuracy = m ? `on ${seasonLabel(m.test_season)}, a season it never saw, it was off by about ${pct(m.results.model.male)}% for a typical player,
    against ${pct(m.results.baseline.male)}% for simply assuming nothing changes.` : "";
  $("panel-body").innerHTML = `
    <div class="sub">about</div>
    <h2>transfer fc</h2>
    <section class="about">
      <p>where premier league players' market values are heading, from how they're playing. a model reads each player's
        minutes, goals, assists, age, club form and value history, and forecasts their transfermarkt value at the end of the season.</p>
      <h3>reading it</h3>
      <p><span class="muted">predicted</span> — the forecast for the end of the season (30 june).<br>
        <span class="muted">actual</span> — the value at the start of the season.<br>
        <span class="muted">↑ ↓</span> — the forecast change between the two. the glow is warm for risers, cool for fallers.<br>
        <span class="muted">why</span> — the factors that pushed the forecast up or down, straight from the model.
        where there's a plain-english version, gemini wrote it from those factors only (and it's checked against them);
        <span class="muted">details</span> underneath shows the exact factors.</p>
      ${live ? `<h3>this season</h3>
      <p>${seasonLabel(live.season)} updates after every gameweek (now after gameweek ${live.gameweek}).
        early on, the forecast mixes this season's pace with last season, so a hot or cold start doesn't dominate.
        forecasts settle as more games are played.</p>` : ""}
      <h3>how good is it</h3>
      <p>${accuracy} it forecasts market values, not transfer fees, and it only knows the stats above —
        not injuries, contracts or transfer rumours.</p>
      <h3>data</h3>
      <p>values and history: transfermarkt, via <a href="https://github.com/dcaribou/transfermarkt-datasets" target="_blank" rel="noopener">transfermarkt-datasets</a>
        (paused at june 2026). this season's stats: the <a href="https://fantasy.premierleague.com" target="_blank" rel="noopener">fantasy premier league</a> feed.
        not affiliated with either.</p>
    </section>`;
  state.current = null;
  setHash("");
  updateSteps();
  showPanel();
}

function closePanel() {
  $("panel").classList.remove("open");
  $("panel").setAttribute("aria-hidden", "true");
  $("scrim").hidden = true;
  state.current = null;
  setHash("");
}

// --- links to a player: #erling-haaland (current season) or #erling-haaland-25-26 --------
const slug = (name) => fold(name).replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
// every player's link, worked out once when the data loads (both directions)
function buildLinks() {
  const latest = state.data.seasons[state.data.seasons.length - 1];
  const counts = new Map();
  for (const p of state.data.players) {
    const k = `${p.season}|${slug(p.name)}`;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  state.links = new Map();
  state.byLink = new Map();
  for (const p of state.data.players) {
    const base = slug(p.name);
    // two players with the same name in one season: add the id to tell them apart
    const name = counts.get(`${p.season}|${base}`) > 1 ? `${base}-${p.id}` : base;
    const link = p.season === latest ? name : `${name}-${seasonLabel(p.season).replace("/", "-")}`;
    state.links.set(p, link);
    state.byLink.set(link, p);
  }
}
const playerHash = (p) => state.links.get(p);
function setHash(hash) {
  const url = hash ? `#${hash}` : location.pathname + location.search;
  if (location.hash !== (hash ? `#${hash}` : "")) history.replaceState(null, "", url);
}
function playerFromHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  if (!hash) return null;
  return state.byLink.get(hash) ?? null;
}
function openFromHash() {
  const p = playerFromHash();
  if (!p) {  // unknown or empty link: don't leave another player's panel showing
    if ($("panel").classList.contains("open") && state.current) closePanel();
    return;
  }
  if (p.season !== state.season) { state.season = p.season; renderSeasons(); renderGrid(); }
  openPanel(p);
}

// --- start -------------------------------------------------------------
function toggleSearch(open) {
  $("search").hidden = !open;
  $("search-toggle").setAttribute("aria-expanded", String(open));
  if (open) {
    $("search-input").focus();
  } else {
    // closing clears the search so the grid goes back to everyone
    $("search-input").value = "";
    if (state.query) { state.query = ""; renderGrid(); }
    $("search-toggle").focus();
  }
}

async function main() {
  loadTheme();
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("menu-toggle").addEventListener("click", () => {
    const open = $("filters").hidden;
    $("filters").hidden = !open;
    $("menu-toggle").setAttribute("aria-expanded", String(open));
  });
  $("panel-close").addEventListener("click", closePanel);
  $("scrim").addEventListener("click", closePanel);
  document.addEventListener("keydown", (e) => {
    const panelOpen = $("panel").getAttribute("aria-hidden") === "false";
    if (panelOpen && state.current && (e.key === "ArrowLeft" || e.key === "ArrowRight") && e.target.tagName !== "INPUT") {
      e.preventDefault();
      step(e.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    if (e.key !== "Escape") return;
    if (panelOpen) closePanel();
    else if (!$("search").hidden) toggleSearch(false);
  });
  $("panel-prev").addEventListener("click", () => step(-1));
  $("panel-next").addEventListener("click", () => step(1));
  $("about-toggle").addEventListener("click", openAbout);
  $("search-toggle").addEventListener("click", () => toggleSearch($("search").hidden));
  $("search-input").addEventListener("input", (e) => {
    state.query = fold(e.target.value.trim());
    renderGrid();
  });
  bindToggleGroup("positions", "group", "group");
  bindToggleGroup("sorts", "sort", "sort");

  try {
    const response = await fetch("data/players.json");
    if (!response.ok) throw new Error(response.status);
    state.data = await response.json();
  } catch (e) {
    $("grid").innerHTML = `<p class="empty">couldn't load the data. try refreshing.</p>`;
    return;
  }
  state.season = state.data.seasons[state.data.seasons.length - 1];
  buildLinks();
  renderSeasons();
  renderGrid();
  openFromHash();
  window.addEventListener("hashchange", openFromHash);
}

main();

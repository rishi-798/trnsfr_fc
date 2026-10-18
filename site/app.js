// transfer fc: browse premier league player-seasons from data/players.json
// (built by scripts/export_site_data.py)

const state = { data: null, season: null, group: "all", sort: "value" };
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
// tiles show the prediction; 22/23 has none (no earlier season to train on), so actual
const shownValue = (p) => p.predicted ?? p.values[3];

function visiblePlayers() {
  let rows = state.data.players.filter((p) => p.season === state.season);
  if (state.group !== "all") rows = rows.filter((p) => p.group === state.group);
  const key = state.sort === "value"
    ? (p) => shownValue(p) ?? -1
    // "all" compares positions by percentile; within one position the raw score works
    : (p) => (state.group === "all" ? p.score_pct : p.score) ?? -1;
  return rows.sort((a, b) => key(b) - key(a));
}

function renderGrid() {
  const rows = visiblePlayers();
  $("count").textContent = `${rows.length} players`;
  $("grid").innerHTML = "";
  rows.forEach((p, i) => {
    const tile = document.createElement("button");
    tile.className = "tile";
    tile.style.animationDelay = `${Math.min(i, 24) * 18}ms`;
    tile.innerHTML = `
      <span class="rank">${String(i + 1).padStart(2, "0")}</span>
      <span class="art"><span class="initials">${p.initials}</span></span>
      <span class="value ${p.predicted == null ? "actual" : ""}">${state.sort === "score" ? rankLabel(p) : money(shownValue(p))}</span>
      <span class="name">${p.name}</span>
      <span class="club">${p.club ?? ""}</span>`;
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
function trendSvg(p) {
  // values are june 30 snapshots: 3y ago, 2y ago, season start, season end
  const years = [p.season - 2, p.season - 1, p.season, p.season + 1];
  const pts = p.values.map((v, i) => ({ v, i, label: `jun ${String(years[i]).slice(2)}` }));
  const known = pts.filter((d) => d.v != null);
  if (known.length < 2) return `<div class="note">not enough history</div>`;
  const max = Math.max(...known.map((d) => d.v)), min = Math.min(...known.map((d) => d.v));
  const x = (i) => 30 + i * 106, y = (v) => 90 - (max === min ? 40 : ((v - min) / (max - min)) * 70);
  const line = known.map((d) => `${x(d.i)},${y(d.v)}`).join(" ");
  return `<svg viewBox="0 0 380 120" preserveAspectRatio="none">
    <polyline class="line" points="${line}"/>
    ${known.map((d) => `<circle class="dot" cx="${x(d.i)}" cy="${y(d.v)}" r="3"/>
      <text x="${x(d.i)}" y="${y(d.v) - 10}" text-anchor="middle">${money(d.v)}</text>`).join("")}
    ${pts.map((d) => `<text x="${x(d.i)}" y="116" text-anchor="middle">${d.label}</text>`).join("")}
  </svg>`;
}

function predictionNote(p) {
  if (p.predicted == null) {
    return p.season === state.data.seasons[0] ? "no earlier season to learn from" : "no starting value to predict from";
  }
  const actual = p.values[3];
  const off = actual ? Math.round(Math.abs(p.predicted / actual - 1) * 100) : null;
  const years = p.trained_on.split("-").map((y) => String(y).slice(2)).join("–");
  return `${off != null ? `${off}% off · ` : ""}learned from ${years}`;
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
  const stale = p.value_end_age_days > 180 ? `<div class="note">value is ${p.value_end_age_days} days old — excluded from training</div>` : "";
  const stats = [
    ["age", p.age], ["apps", p.apps], ["minutes", p.minutes], ["goals", p.goals], ["assists", p.assists],
    ...(p.group === "def" || p.group === "gk" ? [["clean sheets", p.clean_sheets]] : []),
    ["cards", `${p.yellows}y ${p.reds}r`], ["club finish", p.league_position ?? "—"],
    ["champions league", UCL[p.ucl_stage]], ["promoted club", p.promoted ? "yes" : "no"],
    ...(p.other_league_minutes ? [["other league min", p.other_league_minutes]] : []),
    ["last season", prev],
  ];
  $("panel-body").innerHTML = `
    <div class="sub">${seasonLabel(p.season)} · ${p.initials} · ${p.club ?? ""}</div>
    <h2>${p.name}</h2>
    <section class="vs">
      <div><div class="label">predicted${p.explanation ? ` <button class="why" id="why" aria-expanded="false">why ↓</button>` : ""}</div>
        <div class="big ${p.predicted == null ? "pending" : ""}">${money(p.predicted)}</div>
        <div class="note">${predictionNote(p)}</div></div>
      <div><div class="label">actual</div><div class="big">${money(p.values[3])}</div>${stale}</div>
    </section>
    ${p.explanation ? `<p class="explain" id="explain" hidden>${p.explanation}</p>` : ""}
    <section>
      <div class="label">performance score</div>
      <div class="vs"><div class="big">${p.score ?? "—"}</div><div class="big">${p.score_rank ?? "—"}<span class="sub">/${p.group_size}</span></div></div>
      <div class="note">rank among ${p.group_size} ${GROUP_NAMES[p.group]} in ${seasonLabel(p.season)}</div>
    </section>
    <section class="trend"><div class="label">value trend</div>${trendSvg(p)}</section>
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
  $("panel").classList.add("open");
  $("panel").setAttribute("aria-hidden", "false");
  $("scrim").hidden = false;
  $("panel-close").focus();
}

function closePanel() {
  $("panel").classList.remove("open");
  $("panel").setAttribute("aria-hidden", "true");
  $("scrim").hidden = true;
}

// --- start -------------------------------------------------------------
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
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanel(); });
  bindToggleGroup("positions", "group", "group");
  bindToggleGroup("sorts", "sort", "sort");

  state.data = await (await fetch("data/players.json")).json();
  state.season = state.data.seasons[state.data.seasons.length - 1];
  renderSeasons();
  renderGrid();
}

main();

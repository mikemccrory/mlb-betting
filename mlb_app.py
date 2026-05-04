"""
MLB Hit & HR Probability Finder
================================
Data sources:
  - MLB Stats API  : lineups, splits, game logs, pitcher stats
  - Baseball Savant: xBA, xSLG, xwOBA, barrel%, exit velocity
  - wttr.in        : live weather at stadium
Run with:
    streamlit run mlb_app.py
"""

import json
import math
import time
import logging
from io import StringIO
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚾ MLB Probability",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base (desktop + mobile) ─────────────────────────────── */
  [data-testid="stAppViewContainer"] { background:#0f172a; }
  [data-testid="stHeader"]           { background:transparent; }
  h1,h2,h3,h4 { color:#f1f5f9 !important; }

  .stTabs [data-baseweb="tab"] {
    background:#1e293b; border-radius:8px; color:#94a3b8;
    padding:6px 14px; border:1px solid #334155;
  }
  .stTabs [aria-selected="true"] {
    background:#2563eb !important; color:#fff !important; border-color:#2563eb !important;
  }
  /* Scrollable tab list so all tabs fit on narrow screens */
  .stTabs [data-baseweb="tab-list"] {
    overflow-x: auto; flex-wrap: nowrap; gap: 6px;
    -webkit-overflow-scrolling: touch;
  }

  div[data-testid="metric-container"] {
    background:#1e293b; border:1px solid #334155; border-radius:10px; padding:12px 16px;
  }

  .stButton>button {
    background:#1e293b; color:#e2e8f0; border:1px solid #334155;
    border-radius:8px; padding:6px 12px; font-size:13px; line-height:1.5;
    white-space:pre-line; min-height:40px; touch-action:manipulation;
  }
  .stButton>button:hover { background:#2563eb; border-color:#2563eb; color:#fff; }
  .stButton>button[kind="primary"] { background:#2563eb; border-color:#2563eb; color:#fff; }

  label { color:#94a3b8 !important; }
  .stSelectbox>div>div  { background:#1e293b !important; color:#e2e8f0 !important; border-color:#334155 !important; }
  .stTextInput>div>div  { background:#1e293b !important; color:#e2e8f0 !important; border-color:#334155 !important; }
  .stMultiSelect>div>div{ background:#1e293b !important; color:#e2e8f0 !important; border-color:#334155 !important; }
  [data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

  /* Make the main block padding tighter on all screens */
  .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

  /* ── Mobile overrides (≤ 640 px) ─────────────────────────── */
  @media screen and (max-width: 640px) {
    /* Tighten horizontal padding */
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }

    /* Wrap all column blocks so they stack */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.5rem 0 !important; }

    /* Each column: full width by default … */
    [data-testid="column"] { min-width: 100% !important; }

    /* … except the title/refresh row — keep button inline (shrink it) */
    [data-testid="stHorizontalBlock"]:has(h1) [data-testid="column"] {
      min-width: unset !important;
    }

    /* Metric grid: 2-up instead of 4-up */
    div[data-testid="metric-container"] {
      padding: 8px 10px !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="metric-container"]) [data-testid="column"] {
      min-width: calc(50% - 0.25rem) !important;
      flex: 1 1 calc(50% - 0.25rem) !important;
    }

    /* Filters: stack selectbox + input vertically */
    [data-testid="stHorizontalBlock"]:has(.stSelectbox) [data-testid="column"],
    [data-testid="stHorizontalBlock"]:has(.stTextInput)  [data-testid="column"] {
      min-width: 100% !important;
    }

    /* Bigger touch targets on buttons */
    .stButton>button { min-height: 44px !important; font-size: 14px !important; }

    /* Scrollable dataframe */
    [data-testid="stDataFrame"] { overflow-x: auto !important; }

    /* Heading sizes */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MLB_API     = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

LEAGUE_AVG_BA         = 0.248
LEAGUE_AVG_HR_PA      = 0.033
LEAGUE_AVG_BARREL_PCT = 8.0
LEAGUE_AVG_XSLG       = 0.411
LEAGUE_AVG_XBA        = 0.248
LEAGUE_AVG_EXIT_VELO  = 88.5
LEAGUE_AVG_OBP        = 0.320
LEAGUE_AVG_K9         = 8.5
AVG_PA_PER_GAME       = 3.8

# Expected PA by lineup spot (1-indexed); fallback 3.8 for unknown
LINEUP_SPOT_PA = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.1,
                  6: 4.0, 7: 3.9, 8: 3.7, 9: 3.6}

# ML feature order — must match mlb_train_model.py exactly
HIT_FEATURES = ["xba","split_ba","barrel_pct","season_ba",
                 "pitcher_avg_against","pitcher_k_per_9","park_hit_factor",
                 "lineup_spot","pitcher_hand_L"]
HR_FEATURES  = ["xslg","barrel_pct","exit_velo","split_hr_rate",
                 "pitcher_hr_per_9","park_hr_factor","lineup_spot","pitcher_hand_L"]

HR_PARK = {
    "Great American Ball Park":1.28,"Coors Field":1.25,"Citizens Bank Park":1.22,
    "Yankee Stadium":1.20,"Globe Life Field":1.15,"Truist Park":1.12,
    "Camden Yards":1.12,"Fenway Park":1.10,"American Family Field":1.08,
    "Guaranteed Rate Field":1.08,"Wrigley Field":1.08,"Rogers Centre":1.05,
    "Progressive Field":1.05,"Chase Field":1.05,"Minute Maid Park":1.02,
    "Nationals Park":1.03,"Dodger Stadium":1.03,"Busch Stadium":0.97,
    "Target Field":0.97,"Citi Field":0.95,"Tropicana Field":0.90,
    "Kauffman Stadium":0.93,"loanDepot park":0.92,"PNC Park":0.92,
    "T-Mobile Park":0.90,"Comerica Park":0.90,"Petco Park":0.88,
    "Oracle Park":0.85,"Oakland Coliseum":0.88,"Angel Stadium":0.98,
}
HIT_PARK = {
    "Coors Field":1.18,"Fenway Park":1.12,"Great American Ball Park":1.10,
    "Citizens Bank Park":1.08,"Wrigley Field":1.07,"Yankee Stadium":1.05,
    "Rogers Centre":1.05,"Guaranteed Rate Field":1.04,"Camden Yards":1.03,
    "Globe Life Field":1.02,"Truist Park":1.01,"Chase Field":1.00,
    "Minute Maid Park":0.99,"Dodger Stadium":0.98,"American Family Field":0.98,
    "Busch Stadium":0.97,"Target Field":0.96,"Kauffman Stadium":0.96,
    "Citi Field":0.95,"Progressive Field":0.95,"PNC Park":0.95,
    "T-Mobile Park":0.93,"Comerica Park":0.94,"loanDepot park":0.94,
    "Petco Park":0.94,"Oracle Park":0.92,"Tropicana Field":0.90,
}

TEAM_ABBREV = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"OAK","Athletics":"OAK",
    "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SD",
    "San Francisco Giants":"SF","Seattle Mariners":"SEA","St. Louis Cardinals":"STL",
    "Tampa Bay Rays":"TB","Texas Rangers":"TEX","Toronto Blue Jays":"TOR",
    "Washington Nationals":"WSH",
}

VENUE_CITY = {
    "Yankee Stadium":"New York","Citi Field":"New York","Fenway Park":"Boston",
    "Wrigley Field":"Chicago","Guaranteed Rate Field":"Chicago","Dodger Stadium":"Los Angeles",
    "Angel Stadium":"Anaheim","Petco Park":"San Diego","Oracle Park":"San Francisco",
    "T-Mobile Park":"Seattle","Minute Maid Park":"Houston","Globe Life Field":"Arlington",
    "American Family Field":"Milwaukee","Target Field":"Minneapolis","Kauffman Stadium":"Kansas City",
    "Busch Stadium":"St. Louis","Great American Ball Park":"Cincinnati",
    "Progressive Field":"Cleveland","Comerica Park":"Detroit","Truist Park":"Atlanta",
    "Citizens Bank Park":"Philadelphia","Nationals Park":"Washington","PNC Park":"Pittsburgh",
    "loanDepot park":"Miami","Camden Yards":"Baltimore","Tropicana Field":"St. Petersburg",
    "Rogers Centre":"Toronto","Chase Field":"Phoenix","Coors Field":"Denver",
    "Oakland Coliseum":"Oakland",
}

# ── Math helpers ──────────────────────────────────────────────────────────────
def today() -> str: return date.today().isoformat()

def normalize_name(name: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii","ignore").decode("ascii").lower().replace(".","").replace("-"," ").replace("'","").strip()

def _logit(p: float) -> float:
    p = max(1e-6, min(1-1e-6, p))
    return math.log(p / (1 - p))

def _sigmoid(z: float) -> float: return 1 / (1 + math.exp(-z))

def _log_adj(r: float, cap: float = 1.5) -> float:
    return max(-cap, min(cap, math.log(max(r, 1e-4))))

def _safe_float(d, key, fallback=None):
    try:
        v = d.get(key)
        return float(v) if v not in (None, "", "nan", "None") else fallback
    except:
        return fallback

# Short labels for the Key Factors column
_FACTOR_SHORT = {
    "xBA / BA (base)":                  "xBA",
    "Split BA (vs L/R)":                "Split BA",
    "Barrel%":                          "Barrel%",
    "Hot/Cold Streak (L5)":             "L5 form",
    "Form (L10)":                       "L10 form",
    "Park Factor":                      "Park",
    "Temperature":                      "Temp",
    "Pitcher AVG Against (season)":     "Pitcher AVG",
    "Pitcher AVG Against (last 5)":     "Pitcher L5",
    "Base (league avg HR rate)":        "Base",
    "HR/AB Split (vs L/R)":             "HR split",
    "Exit Velocity":                    "Exit velo",
    "xSLG":                             "xSLG",
    "HR Rate (L5)":                     "L5 HR",
    "HR Rate (L10)":                    "L10 HR",
    "Park Factor (HR)":                 "Park",
    "Pitcher HR/9 (season)":            "P-HR/9",
    "Pitcher ERA":                      "P-ERA",
    "Pitcher HR/9 (last 5)":            "P-HR/9 L5",
}

def top_factors(hit_c: Dict, hr_c: Dict, n: int = 3) -> str:
    """
    Build a brief 'Key Factors' string for the table.
    Shows top n boosters and top n detractors across both models,
    deduplicating shared factors.
    """
    merged: Dict[str, float] = {}
    for c in (hit_c, hr_c):
        for k, v in c.items():
            if k.startswith("_") or "base" in k.lower():
                continue
            short = _FACTOR_SHORT.get(k, k)
            # take the stronger signal if the same label appears in both
            if short not in merged or abs(v) > abs(merged[short]):
                merged[short] = v

    if not merged:
        return "—"

    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    boosters  = [(lbl, v) for lbl, v in sorted_items if v >  0.02][:n]
    detractors= [(lbl, v) for lbl, v in sorted_items if v < -0.02][-n:]

    lines = []
    if boosters:
        lines.append("↑ " + ", ".join(lbl for lbl, _ in boosters))
    if detractors:
        lines.append("↓ " + ", ".join(lbl for lbl, _ in reversed(detractors)))
    return "\n".join(lines) if lines else "—"

def _temp_hit_adj(f: float) -> float:
    if f < 40: return -0.20
    if f < 50: return -0.12
    if f < 60: return -0.06
    if f < 68: return -0.02
    if f <= 85: return 0.0
    return 0.02

def _temp_hr_adj(f: float) -> float:
    if f < 40: return -0.25
    if f < 50: return -0.15
    if f < 60: return -0.08
    if f < 68: return -0.03
    if f <= 85: return 0.0
    return 0.06

# ── Data fetching ─────────────────────────────────────────────────────────────
def _get(url: str, params=None, timeout=15) -> requests.Response:
    return requests.get(url, params=params, timeout=timeout)

@st.cache_data(ttl=60, show_spinner=False)
def get_live_scores(date_str: str) -> List[Dict]:
    try:
        r = _get(f"{MLB_API}/schedule", {"sportId":1,"date":date_str,"hydrate":"linescore,team"})
        r.raise_for_status()
        out = []
        for d in r.json().get("dates",[]):
            for g in d.get("games",[]):
                ls  = g.get("linescore",{})
                st_ = g.get("status",{})
                home = g["teams"]["home"]["team"]["name"]
                away = g["teams"]["away"]["team"]["name"]
                out.append({
                    "game_id":     g["gamePk"],
                    "home":        home,
                    "away":        away,
                    "home_abbrev": TEAM_ABBREV.get(home, home[:3].upper()),
                    "away_abbrev": TEAM_ABBREV.get(away, away[:3].upper()),
                    "home_score":  ls.get("teams",{}).get("home",{}).get("runs",0),
                    "away_score":  ls.get("teams",{}).get("away",{}).get("runs",0),
                    "inning":      ls.get("currentInning",0),
                    "inning_half": ls.get("inningHalf",""),
                    "state":       st_.get("abstractGameState","Preview"),
                    "game_time":   g.get("gameDate",""),
                })
        return out
    except Exception as e:
        log.warning(f"Live scores: {e}")
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_team_roster(team_id: int) -> List[Dict]:
    try:
        r = _get(f"{MLB_API}/teams/{team_id}/roster",
                 {"rosterType":"active","season":date.today().year})
        r.raise_for_status()
        return [{"id":p["person"]["id"],"name":p["person"]["fullName"],"confirmed":False}
                for p in r.json().get("roster",[])
                if p.get("position",{}).get("abbreviation","P") not in ("P","SP","RP","CL")]
    except:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_todays_games(date_str: str) -> List[Dict]:
    r = _get(f"{MLB_API}/schedule",{"sportId":1,"date":date_str,
             "hydrate":"lineups,probablePitcher(note),team,venue"})
    r.raise_for_status()
    games = []
    for d in r.json().get("dates",[]):
        for g in d.get("games",[]):
            info = {
                "game_id":       g["gamePk"],
                "game_time":     g.get("gameDate",""),
                "venue":         g.get("venue",{}).get("name","Unknown Ballpark"),
                "home_team":     g["teams"]["home"]["team"]["name"],
                "away_team":     g["teams"]["away"]["team"]["name"],
                "home_team_id":  g["teams"]["home"]["team"]["id"],
                "away_team_id":  g["teams"]["away"]["team"]["id"],
            }
            for side in ("home","away"):
                pp = g["teams"][side].get("probablePitcher",{})
                info[f"{side}_pitcher_id"]   = pp.get("id")
                info[f"{side}_pitcher_name"] = pp.get("fullName","TBD")
            lineups = g.get("lineups",{})
            for side in ("home","away"):
                players = lineups.get(f"{side}Players",[])
                info[f"{side}_lineup"] = [
                    {"id":p["id"],"name":p.get("fullName",""),"confirmed":True}
                    for p in players
                ]
            games.append(info)
    for game in games:
        for side in ("home","away"):
            if not game[f"{side}_lineup"]:
                game[f"{side}_lineup"] = get_team_roster(game[f"{side}_team_id"])
    return games

@st.cache_data(ttl=300, show_spinner=False)
def get_savant_data(year: int) -> pd.DataFrame:
    try:
        r = requests.get(f"{SAVANT_BASE}/expected_statistics", params={
            "type":"batter","year":year,"position":"","team":"","min":25,"csv":"true"
        }, headers={"User-Agent":"Mozilla/5.0"}, timeout=25)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        col_map = {
            "player_id":"mlb_id","est_ba":"xba","est_slg":"xslg","est_woba":"xwoba",
            "barrel_batted_rate":"barrel_pct","exit_velocity_avg":"exit_velo",
            "avg_hit_speed":"exit_velo","launch_angle_avg":"launch_angle",
            "ba":"ba","slg":"slg","on_base_percent":"obp",
        }
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        if "mlb_id" not in df.columns:
            return pd.DataFrame()
        df["mlb_id"] = df["mlb_id"].astype(str)
        return df.set_index("mlb_id")
    except Exception as e:
        log.warning(f"Savant failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_player_splits(player_id: int, year: int) -> Dict:
    try:
        r = _get(f"{MLB_API}/people/{player_id}/stats",
                 {"stats":"statSplits","season":year,"group":"hitting","sitCodes":"vl,vr"})
        r.raise_for_status()
        splits = {}
        for sp in r.json().get("stats",[{}])[0].get("splits",[]):
            code = sp.get("split",{}).get("code","")
            s    = sp.get("stat",{})
            pa   = int(s.get("plateAppearances",0) or 0)
            hr   = int(s.get("homeRuns",0) or 0)
            ab   = int(s.get("atBats",0) or 0)
            splits[code] = {
                "avg": float(s.get("avg",0) or 0),
                "slg": float(s.get("slg",0) or 0),
                "ops": float(s.get("ops",0) or 0),
                "obp": float(s.get("obp",0) or 0),
                "hr":hr,"pa":pa,"ab":ab,
                "hr_rate": hr/max(ab,1),
            }
        return splits
    except:
        return {}

@st.cache_data(ttl=300, show_spinner=False)
def get_player_game_log(player_id: int, year: int, n: int = 10) -> List[Dict]:
    try:
        r = _get(f"{MLB_API}/people/{player_id}/stats",
                 {"stats":"gameLog","season":year,"group":"hitting"})
        r.raise_for_status()
        raw = sorted(r.json().get("stats",[{}])[0].get("splits",[]),
                     key=lambda x: x.get("date",""), reverse=True)[:n]
        return [{
            "date": g.get("date",""),
            "opp":  g.get("opponent",{}).get("abbreviation",""),
            "ab":   g["stat"].get("atBats",0),
            "h":    g["stat"].get("hits",0),
            "hr":   g["stat"].get("homeRuns",0),
            "rbi":  g["stat"].get("rbi",0),
            "bb":   g["stat"].get("baseOnBalls",0),
            "k":    g["stat"].get("strikeOuts",0),
            "2b":   g["stat"].get("doubles",0),
            "3b":   g["stat"].get("triples",0),
        } for g in raw]
    except:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_pitcher_hand(pitcher_id: Optional[int]) -> str:
    if not pitcher_id: return "R"
    try:
        r = _get(f"{MLB_API}/people/{pitcher_id}")
        r.raise_for_status()
        return r.json().get("people",[{}])[0].get("pitchHand",{}).get("code","R") or "R"
    except:
        return "R"

@st.cache_data(ttl=300, show_spinner=False)
def get_pitcher_stats(pitcher_id: Optional[int], year: int) -> Dict:
    base = {"era":4.50,"whip":1.30,"hr_per_9":1.20,"avg_against":0.250,
            "k_per_9":8.0,"bb_per_9":3.0,"l5_avg_against":None,"l5_hr_per_9":None,
            "l5_era":None}
    if not pitcher_id: return base
    try:
        r = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                 {"stats":"season","season":year,"group":"pitching"})
        r.raise_for_status()
        sp = r.json().get("stats",[{}])[0].get("splits",[])
        if sp:
            s = sp[0].get("stat",{})
            base.update({
                "era":          float(s.get("era",4.50) or 4.50),
                "whip":         float(s.get("whip",1.30) or 1.30),
                "hr_per_9":     float(s.get("homeRunsPer9",1.20) or 1.20),
                "avg_against":  float(s.get("avg",0.250) or 0.250),
                "k_per_9":      float(s.get("strikeoutsPer9Inn",8.0) or 8.0),
                "bb_per_9":     float(s.get("walksPer9Inn",3.0) or 3.0),
            })
        # Last 5 game log
        rl = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                  {"stats":"gameLog","season":year,"group":"pitching"})
        rl.raise_for_status()
        gl = sorted(rl.json().get("stats",[{}])[0].get("splits",[]),
                    key=lambda x: x.get("date",""), reverse=True)[:5]
        if gl:
            t_ab = sum(int(g["stat"].get("atBats",0) or 0) for g in gl)
            t_h  = sum(int(g["stat"].get("hits",0) or 0) for g in gl)
            t_hr = sum(int(g["stat"].get("homeRuns",0) or 0) for g in gl)
            t_ip = sum(float(g["stat"].get("inningsPitched",0) or 0) for g in gl)
            t_er = sum(int(g["stat"].get("earnedRuns",0) or 0) for g in gl)
            if t_ab >= 10:
                base["l5_avg_against"] = round(t_h/t_ab, 3)
            if t_ip >= 5:
                base["l5_hr_per_9"]    = round(t_hr/t_ip*9, 2)
                base["l5_era"]         = round(t_er/t_ip*9, 2)
        return base
    except:
        return base

@st.cache_data(ttl=300, show_spinner=False)
def get_team_obp(team_id: int, year: int) -> float:
    try:
        r = _get(f"{MLB_API}/teams/{team_id}/stats",
                 {"stats":"season","season":year,"group":"hitting"})
        r.raise_for_status()
        sp = r.json().get("stats",[{}])[0].get("splits",[])
        if sp:
            return float(sp[0].get("stat",{}).get("obp", LEAGUE_AVG_OBP) or LEAGUE_AVG_OBP)
    except:
        pass
    return LEAGUE_AVG_OBP

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(venue: str) -> Dict:
    city = VENUE_CITY.get(venue)
    if not city:
        return {"temp_f":72,"wind_mph":5,"condition":"Unknown","available":False}
    try:
        r = requests.get(f"https://wttr.in/{city.replace(' ','+')}?format=j1",
                         timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        c = r.json()["current_condition"][0]
        return {
            "temp_f":       int(c.get("temp_F",72)),
            "feels_like_f": int(c.get("FeelsLikeF",72)),
            "wind_mph":     int(c.get("windspeedMiles",5)),
            "wind_dir":     c.get("winddir16Point","N"),
            "condition":    c.get("weatherDesc",[{}])[0].get("value","Unknown"),
            "available":    True,
        }
    except Exception as e:
        log.warning(f"Weather {venue}: {e}")
        return {"temp_f":72,"wind_mph":5,"condition":"Unknown","available":False}

# ── ML model loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ml_models():
    """
    Load trained joblib models from mlb_models/.
    Returns (hit_model, hr_model, model_info) or (None, None, None) if not found.
    """
    try:
        import joblib
        model_dir = Path(__file__).parent / "mlb_models"
        hit_model  = joblib.load(model_dir / "hit_model.joblib")
        hr_model   = joblib.load(model_dir / "hr_model.joblib")
        model_info = json.loads((model_dir / "model_info.json").read_text())
        log.info("Loaded trained ML models from mlb_models/")
        return hit_model, hr_model, model_info
    except Exception as e:
        log.info(f"No trained models found ({e}); using rule-based fallback")
        return None, None, None


def _ml_contributions(lr_info: Dict, raw_values: List[Optional[float]]) -> Dict[str, float]:
    """
    Compute per-feature logit contributions from a trained LR.
    contribution_i = coef_i * (x_i - scaler_mean_i) / scaler_scale_i
    Missing features (None / NaN) fall back to imputer mean → contribution ≈ 0.
    """
    means   = lr_info["imputer_means"]
    sc_mean = lr_info["scaler_means"]
    sc_scl  = lr_info["scaler_scales"]
    coefs   = lr_info["lr_coef"]
    features= lr_info["features"]

    contribs = {}
    for i, (feat, coef) in enumerate(zip(features, coefs)):
        x = raw_values[i]
        if x is None or (isinstance(x, float) and math.isnan(x)):
            x = means[i]  # imputer mean → contrib ≈ 0
        scaled = (x - sc_mean[i]) / max(sc_scl[i], 1e-8)
        contribs[feat] = round(coef * scaled, 4)
    return contribs


def _build_hit_raw(splits, pitcher_hand, venue, pitcher_stats, savant, lineup_spot) -> List:
    split       = splits.get("vl" if pitcher_hand == "L" else "vr", {})
    split_ba    = _safe_float(split, "avg")
    split_hr_ab = split.get("hr", 0) / max(split.get("ab", 1), 1)
    return [
        _safe_float(savant, "xba"),
        split_ba,
        _safe_float(savant, "barrel_pct"),
        _safe_float(savant, "ba"),
        pitcher_stats.get("avg_against", LEAGUE_AVG_BA),
        pitcher_stats.get("k_per_9", LEAGUE_AVG_K9),
        HIT_PARK.get(venue, 1.0),
        float(lineup_spot) if lineup_spot else 5.0,   # fallback to middle of order
        1.0 if pitcher_hand == "L" else 0.0,
    ]


def _build_hr_raw(splits, pitcher_hand, venue, pitcher_stats, savant, lineup_spot) -> List:
    split      = splits.get("vl" if pitcher_hand == "L" else "vr", {})
    split_hr_r = split.get("hr", 0) / max(split.get("ab", 1), 1) if split.get("ab", 0) >= 20 else None
    return [
        _safe_float(savant, "xslg"),
        _safe_float(savant, "barrel_pct"),
        _safe_float(savant, "exit_velo"),
        split_hr_r,
        pitcher_stats.get("hr_per_9", 1.20),
        HR_PARK.get(venue, 1.0),
        float(lineup_spot) if lineup_spot else 5.0,
        1.0 if pitcher_hand == "L" else 0.0,
    ]


def compute_hit_prob_ml(
    hit_model, hr_model, model_info: Dict,
    splits: Dict, pitcher_hand: str,
    venue: str, pitcher_stats: Dict, savant: Dict,
    lineup_spot: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """ML-based game-level hit probability. Returns (p_game, contributions)."""
    raw   = _build_hit_raw(splits, pitcher_hand, venue, pitcher_stats, savant, lineup_spot)
    X     = np.array(raw, dtype=object).reshape(1, -1)
    p_game = float(hit_model.predict_proba(X)[0][1])
    p_game = round(max(0.01, min(0.99, p_game)), 4)

    contribs = _ml_contributions(model_info["hit_lr_info"], raw)
    contribs["_p_per_pa"] = p_game  # game-level; field kept for compatibility
    contribs["_est_pa"]   = float(LINEUP_SPOT_PA.get(lineup_spot, AVG_PA_PER_GAME))
    return p_game, contribs


def compute_hr_prob_ml(
    hit_model, hr_model, model_info: Dict,
    splits: Dict, pitcher_hand: str,
    venue: str, pitcher_stats: Dict, savant: Dict,
    lineup_spot: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """ML-based game-level HR probability. Returns (p_game, contributions)."""
    raw   = _build_hr_raw(splits, pitcher_hand, venue, pitcher_stats, savant, lineup_spot)
    X     = np.array(raw, dtype=object).reshape(1, -1)
    p_game = float(hr_model.predict_proba(X)[0][1])
    p_game = round(max(0.001, min(0.99, p_game)), 4)

    contribs = _ml_contributions(model_info["hr_lr_info"], raw)
    contribs["_p_per_pa"] = p_game
    contribs["_est_pa"]   = float(LINEUP_SPOT_PA.get(lineup_spot, AVG_PA_PER_GAME))
    return p_game, contribs


# ── Probability models (rule-based fallback) ───────────────────────────────────
def _recent_avg(games: List[Dict], n: int) -> Optional[float]:
    g = games[:n]; ab = sum(x["ab"] for x in g); h = sum(x["h"] for x in g)
    return (h/ab) if ab >= 5 else None

def _recent_hr_rate(games: List[Dict], n: int) -> Optional[float]:
    g = games[:n]; ab = sum(x["ab"] for x in g); hr = sum(x["hr"] for x in g)
    return (hr/ab) if ab >= 5 else None

def compute_hit_prob(
    splits: Dict, pitcher_hand: str, game_log: List[Dict],
    venue: str, pitcher_stats: Dict, savant: Dict,
    team_obp: float = LEAGUE_AVG_OBP, temp_f: float = 72,
    lineup_spot: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Logistic regression for P(at least 1 hit in game).
    Hitter skill features carry ~3x more total weight than pitcher features.
    """
    split = splits.get("vl" if pitcher_hand == "L" else "vr", {})
    contribs: Dict[str, float] = {}

    # ── BASE: xBA → xBA is the single best predictor of true hit rate ──────
    xba      = _safe_float(savant, "xba")
    savant_ba= _safe_float(savant, "ba")
    split_ba = _safe_float(split,  "avg")
    base = xba or savant_ba or split_ba or LEAGUE_AVG_XBA
    base = max(0.08, min(0.44, base))
    z    = _logit(base)
    # base sets the prior — log that for the waterfall
    contribs["xBA / BA (base)"] = round(z - _logit(LEAGUE_AVG_XBA), 4)

    # ── HITTER: Split BA vs this pitcher handedness ──────────────────────────
    # Weight 1.4 — most differentiating feature per-player
    if split_ba and split.get("pa",0) >= 30:
        adj = 1.4 * _log_adj(split_ba / LEAGUE_AVG_BA)
        z += adj; contribs["Split BA (vs L/R)"] = round(adj, 4)

    # ── HITTER: Barrel% — contact quality ────────────────────────────────────
    barrel = _safe_float(savant, "barrel_pct")
    if barrel and barrel > 0:
        adj = 0.5 * _log_adj(barrel / LEAGUE_AVG_BARREL_PCT, 1.2)
        z += adj; contribs["Barrel%"] = round(adj, 4)

    # ── HITTER: Recent form L5 ────────────────────────────────────────────────
    a5 = _recent_avg(game_log, 5)
    if a5 is not None:
        adj = 1.1 * _log_adj(a5 / LEAGUE_AVG_BA)
        z += adj; contribs["Hot/Cold Streak (L5)"] = round(adj, 4)

    # ── HITTER: Recent form L10 ───────────────────────────────────────────────
    a10 = _recent_avg(game_log, 10)
    if a10 is not None:
        adj = 0.55 * _log_adj(a10 / LEAGUE_AVG_BA, 1.3)
        z += adj; contribs["Form (L10)"] = round(adj, 4)

    # ── PARK: hit factor ──────────────────────────────────────────────────────
    pf  = HIT_PARK.get(venue, 1.0)
    adj = 0.4 * _log_adj(pf, 0.6)
    z  += adj; contribs["Park Factor"] = round(adj, 4)

    # ── PITCHER: season opp BA ────────────────────────────────────────────────
    p_ba = pitcher_stats.get("avg_against", LEAGUE_AVG_BA)
    adj  = 0.65 * _log_adj(p_ba / LEAGUE_AVG_BA, 1.2)
    z   += adj; contribs["Pitcher AVG Against (season)"] = round(adj, 4)

    # ── PITCHER: last 5 starts opp BA ────────────────────────────────────────
    p_l5 = pitcher_stats.get("l5_avg_against")
    if p_l5 is not None:
        adj = 0.70 * _log_adj(p_l5 / LEAGUE_AVG_BA, 1.2)
        z  += adj; contribs["Pitcher AVG Against (last 5)"] = round(adj, 4)

    # ── PITCHER: K rate — higher K rate suppresses hits ───────────────────────
    p_k9 = pitcher_stats.get("k_per_9", LEAGUE_AVG_K9)
    adj  = -0.45 * _log_adj(p_k9 / LEAGUE_AVG_K9, 1.0)
    z   += adj; contribs["Pitcher K Rate"] = round(adj, 4)

    # ── OPPORTUNITIES: lineup spot → estimated PA ─────────────────────────────
    if lineup_spot in LINEUP_SPOT_PA:
        est_pa = LINEUP_SPOT_PA[lineup_spot]
    else:
        est_pa = max(2.8, min(5.0, AVG_PA_PER_GAME + (team_obp - LEAGUE_AVG_OBP) * 4))

    p_pa   = _sigmoid(z)
    p_pa   = max(0.03, min(0.55, p_pa))   # loosened bounds — elite hitters can separate
    p_game = round(max(0.04, 1 - (1 - p_pa) ** est_pa), 4)

    contribs["_p_per_pa"] = round(p_pa, 4)
    contribs["_est_pa"]   = round(est_pa, 2)
    return p_game, contribs


def compute_hr_prob(
    splits: Dict, pitcher_hand: str, game_log: List[Dict],
    venue: str, pitcher_stats: Dict, savant: Dict,
    team_obp: float = LEAGUE_AVG_OBP, temp_f: float = 72,
    lineup_spot: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Logistic regression for P(at least 1 HR in game).
    Heavily weighted toward individual hitter power metrics.
    """
    split = splits.get("vl" if pitcher_hand == "L" else "vr", {})
    contribs: Dict[str, float] = {}

    z = _logit(LEAGUE_AVG_HR_PA)
    contribs["Base (league avg HR rate)"] = 0.0

    # ── HITTER: split HR/AB vs this handedness ────────────────────────────────
    sp_ab = int(split.get("ab",0) or 0)
    sp_hr = int(split.get("hr",0) or 0)
    if sp_ab >= 30:
        rate = sp_hr / sp_ab
        adj  = 1.6 * _log_adj((rate + LEAGUE_AVG_HR_PA) / (2 * LEAGUE_AVG_HR_PA), 1.5)
        z   += adj; contribs["HR/AB Split (vs L/R)"] = round(adj, 4)

    # ── HITTER: Barrel% ───────────────────────────────────────────────────────
    barrel = _safe_float(savant, "barrel_pct")
    if barrel and barrel > 0:
        adj = 1.8 * _log_adj(barrel / LEAGUE_AVG_BARREL_PCT, 1.5)
        z  += adj; contribs["Barrel%"] = round(adj, 4)

    # ── HITTER: Exit velocity ─────────────────────────────────────────────────
    ev = _safe_float(savant, "exit_velo")
    if ev and ev > 50:
        adj = 1.0 * _log_adj(ev / LEAGUE_AVG_EXIT_VELO, 1.2)
        z  += adj; contribs["Exit Velocity"] = round(adj, 4)

    # ── HITTER: xSLG ─────────────────────────────────────────────────────────
    xslg = _safe_float(savant, "xslg")
    if xslg:
        adj = 0.8 * _log_adj(xslg / LEAGUE_AVG_XSLG, 1.2)
        z  += adj; contribs["xSLG"] = round(adj, 4)

    # ── HITTER: Recent HR rate L5 — dampened; 5 games is tiny sample ────────
    h5 = _recent_hr_rate(game_log, 5)
    if h5 is not None:
        adj = 0.45 * _log_adj((h5 + LEAGUE_AVG_HR_PA) / (2 * LEAGUE_AVG_HR_PA), 1.0)
        z  += adj; contribs["HR Rate (L5)"] = round(adj, 4)

    # ── HITTER: Recent HR rate L10 ────────────────────────────────────────────
    h10 = _recent_hr_rate(game_log, 10)
    if h10 is not None:
        adj = 0.6 * _log_adj((h10 + LEAGUE_AVG_HR_PA) / (2 * LEAGUE_AVG_HR_PA), 1.2)
        z  += adj; contribs["HR Rate (L10)"] = round(adj, 4)

    # ── PARK: HR factor ───────────────────────────────────────────────────────
    pf  = HR_PARK.get(venue, 1.0)
    adj = 0.7 * _log_adj(pf, 0.8)
    z  += adj; contribs["Park Factor (HR)"] = round(adj, 4)

    # ── PITCHER: HR/9 (season) ────────────────────────────────────────────────
    p_hr9 = pitcher_stats.get("hr_per_9", 1.20)
    adj   = 0.7 * _log_adj(p_hr9 / 1.20, 1.1)
    z    += adj; contribs["Pitcher HR/9 (season)"] = round(adj, 4)

    # ── PITCHER: ERA ─────────────────────────────────────────────────────────
    era = pitcher_stats.get("era", 4.50)
    adj = 0.3 * _log_adj(era / 4.50, 0.8)
    z  += adj; contribs["Pitcher ERA"] = round(adj, 4)

    # ── PITCHER: HR rate last 5 starts ────────────────────────────────────────
    p_l5_hr9 = pitcher_stats.get("l5_hr_per_9")
    if p_l5_hr9 is not None:
        adj = 0.55 * _log_adj(p_l5_hr9 / 1.20, 1.1)
        z  += adj; contribs["Pitcher HR/9 (last 5)"] = round(adj, 4)

    # ── OPPORTUNITIES: lineup spot → estimated PA ─────────────────────────────
    if lineup_spot in LINEUP_SPOT_PA:
        est_pa = LINEUP_SPOT_PA[lineup_spot]
    else:
        est_pa = max(2.8, min(5.0, AVG_PA_PER_GAME + (team_obp - LEAGUE_AVG_OBP) * 4))

    p_pa   = _sigmoid(z)
    p_pa   = max(0.002, min(0.18, p_pa))  # loosened ceiling; power hitters can reach ~57% game prob
    p_game = round(max(0.002, 1 - (1 - p_pa) ** est_pa), 4)

    contribs["_p_per_pa"] = round(p_pa, 4)
    contribs["_est_pa"]   = round(est_pa, 2)
    return p_game, contribs

# ── Main analysis (cached daily) ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(date_str: str) -> Dict:
    games = get_todays_games(date_str)
    if not games:
        return {"projections":[], "games":[], "model":"rule-based"}

    year       = date.today().year
    savant_df  = get_savant_data(year)
    projections: List[Dict] = []
    seen: set = set()

    # Try trained ML models; fall back to rule-based if not yet trained
    hit_model, hr_model, model_info = load_ml_models()
    using_ml = hit_model is not None

    for game in games:
        venue   = game["venue"]
        weather = get_weather(venue)
        temp_f  = weather.get("temp_f", 72)
        home_obp = get_team_obp(game["home_team_id"], year)
        away_obp = get_team_obp(game["away_team_id"], year)

        for side in ("home","away"):
            opp      = "away" if side=="home" else "home"
            opp_pid  = game.get(f"{opp}_pitcher_id")
            opp_pnm  = game.get(f"{opp}_pitcher_name","TBD")
            lineup   = game.get(f"{side}_lineup",[])
            team_nm  = game.get(f"{side}_team","")
            team_obp = home_obp if side=="home" else away_obp
            if not lineup: continue

            p_hand  = get_pitcher_hand(opp_pid)
            p_stats = get_pitcher_stats(opp_pid, year)

            for lineup_idx, player in enumerate(lineup):
                pid = player["id"]; name = player["name"]
                if pid in seen: continue
                seen.add(pid)

                # 1-indexed lineup spot (0 if not in confirmed lineup)
                lineup_spot = lineup_idx + 1 if player.get("confirmed") else 0

                splits = get_player_splits(pid, year)
                gl     = get_player_game_log(pid, year, 10)
                time.sleep(0.02)

                xstats: Dict = {}
                pid_s = str(pid)
                if pid_s in savant_df.index:
                    row = savant_df.loc[pid_s]
                    for col in ["xba","xslg","xwoba","barrel_pct","exit_velo","ba","slg","launch_angle"]:
                        v = row.get(col)
                        if v is not None and pd.notna(v):
                            xstats[col] = str(v)

                if using_ml:
                    p_hit, hit_c = compute_hit_prob_ml(
                        hit_model, hr_model, model_info,
                        splits, p_hand, venue, p_stats, xstats, lineup_spot)
                    p_hr,  hr_c  = compute_hr_prob_ml(
                        hit_model, hr_model, model_info,
                        splits, p_hand, venue, p_stats, xstats, lineup_spot)
                else:
                    p_hit, hit_c = compute_hit_prob(splits, p_hand, gl, venue, p_stats, xstats, team_obp, temp_f, lineup_spot)
                    p_hr,  hr_c  = compute_hr_prob( splits, p_hand, gl, venue, p_stats, xstats, team_obp, temp_f, lineup_spot)

                projections.append({
                    "player":         name,
                    "team":           team_nm,
                    "game_id":        game["game_id"],
                    "home_team":      game["home_team"],
                    "away_team":      game["away_team"],
                    "venue":          venue,
                    "pitcher":        opp_pnm,
                    "pitcher_hand":   p_hand,
                    "pitcher_stats":  p_stats,
                    "p_hit":          p_hit,
                    "p_hr":           p_hr,
                    "hit_contributions": hit_c,
                    "hr_contributions":  hr_c,
                    "xstats":         xstats,
                    "splits":         splits,
                    "game_log":       gl,
                    "confirmed":      player.get("confirmed", False),
                    "weather":        weather,
                    "team_obp":       team_obp,
                    "lineup_spot":    lineup_spot,
                })

    projections.sort(key=lambda x: x["p_hit"], reverse=True)
    return {"projections": projections, "games": games,
            "model": "ml" if using_ml else "rule-based",
            "model_info": model_info}

# ── Chart helpers ─────────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font_color="#94a3b8")

def waterfall_chart(contribs: Dict, title: str) -> go.Figure:
    items = {k:v for k,v in contribs.items()
             if not k.startswith("_") and v != 0 and "base" not in k.lower()}
    if not items:
        return go.Figure()
    labels = list(items.keys())
    values = list(items.values())
    colors = ["#34d399" if v > 0 else "#f87171" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{'+'if v>0 else ''}{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **DARK, title_text=title, title_font_color="#64748b", title_font_size=12,
        height=max(200, 32*len(labels)+60),
        margin=dict(l=10, r=70, t=36, b=10),
        xaxis=dict(showgrid=True, gridcolor="#334155", zeroline=True, zerolinecolor="#475569"),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return fig

def advanced_stats_chart(xstats: Dict, splits: Dict, pitcher_hand: str) -> go.Figure:
    split = splits.get("vl" if pitcher_hand=="L" else "vr", {})
    xba   = _safe_float(xstats,"xba")
    ba    = _safe_float(xstats,"ba") or _safe_float(split,"avg")
    xslg  = _safe_float(xstats,"xslg")
    slg   = _safe_float(xstats,"slg") or _safe_float(split,"slg")
    xwoba = _safe_float(xstats,"xwoba")

    cats, p_vals, lg_vals = [], [], []
    for label, pv, lv in [
        ("BA", ba, 0.248), ("xBA", xba, 0.248),
        ("SLG", slg, 0.411), ("xSLG", xslg, 0.411),
        ("xwOBA", xwoba, 0.315),
    ]:
        if pv is not None:
            cats.append(label); p_vals.append(pv); lg_vals.append(lv)

    if not cats:
        return go.Figure()

    colors = ["#34d399" if p >= l else "#f87171" for p,l in zip(p_vals, lg_vals)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Player", x=cats, y=p_vals, marker_color=colors,
        text=[f"{v:.3f}" for v in p_vals], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        name="Lg Avg", x=cats, y=lg_vals, mode="markers",
        marker=dict(symbol="line-ew-open", size=18, color="#94a3b8", line=dict(width=3)),
    ))
    fig.update_layout(
        **DARK, height=260, margin=dict(l=0,r=0,t=10,b=0),
        barmode="group", legend=dict(font_color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="#334155"),
    )
    return fig

def game_log_chart(gl: List[Dict]) -> go.Figure:
    if not gl: return go.Figure()
    df = pd.DataFrame(gl[::-1])
    df["label"] = df["date"].str[5:] + " " + df["opp"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Hits", x=df["label"], y=df["h"],
                         marker_color="#34d399", text=df["h"], textposition="outside"))
    fig.add_trace(go.Bar(name="HR",   x=df["label"], y=df["hr"],
                         marker_color="#f97316", text=df["hr"], textposition="outside"))
    fig.update_layout(
        **DARK, height=220, margin=dict(l=0,r=0,t=10,b=55),
        barmode="group", legend=dict(font_color="#94a3b8"),
        xaxis=dict(tickangle=-35, tickfont_size=10),
        yaxis=dict(showgrid=True, gridcolor="#334155"),
    )
    return fig

# ── Player modal ──────────────────────────────────────────────────────────────
@st.dialog("Player Detail", width="large")
def show_player_modal(player_name: str, projections: List[Dict]):
    p = next((x for x in projections if x["player"] == player_name), None)
    if not p:
        st.error("Player not found.")
        return

    abbrev    = TEAM_ABBREV.get(p["team"], p["team"][:3])
    hand_lbl  = "⬅ LHP" if p["pitcher_hand"]=="L" else "➡ RHP"
    spot      = p.get("lineup_spot", 0)
    spot_lbl  = f" · Bat #{spot}" if spot else ""
    st.markdown(f"## {p['player']}")
    st.caption(f"**{abbrev}** · vs **{p['pitcher']}** ({hand_lbl}) · {p['venue']}{spot_lbl}")

    w = p.get("weather",{})
    if w.get("available"):
        st.caption(f"🌡 {w['temp_f']}°F · 💨 {w['wind_mph']} mph {w.get('wind_dir','')} · {w.get('condition','')}")

    c1, c2, c3 = st.columns(3)
    c1.metric("P(Hit in Game)", f"{p['p_hit']*100:.1f}%")
    c2.metric("P(HR in Game)",  f"{p['p_hr']*100:.1f}%")
    spot   = p.get("lineup_spot", 0)
    est_pa = LINEUP_SPOT_PA.get(spot, p["hit_contributions"].get("_est_pa", 3.8))
    spot_s = f" (#{spot})" if spot else ""
    c3.metric("Est. PA Today",  f"{est_pa:.1f}{spot_s}")

    st.divider()
    tab_stats, tab_form, tab_breakdown = st.tabs(["Advanced Stats", "Last 10 Games", "Probability Breakdown"])

    with tab_stats:
        xs    = p.get("xstats",{})
        splits= p.get("splits",{})
        ph    = p["pitcher_hand"]
        split = splits.get("vl" if ph=="L" else "vr", {})

        def _fmt(d, k, fmt=".3f"):
            v = _safe_float(d, k)
            return format(v, fmt) if v is not None else "—"

        mc = st.columns(5)
        ba_val   = _fmt(xs,"ba") if _fmt(xs,"ba")!="—" else _fmt(split,"avg")
        barrel_v = (_fmt(xs,"barrel_pct",".1f")+"%" if _fmt(xs,"barrel_pct",".1f")!="—" else "—")
        ev_v     = (_fmt(xs,"exit_velo",".1f")+" mph" if _fmt(xs,"exit_velo",".1f")!="—" else "—")
        mc[0].metric("BA",       ba_val,           delta="lg .248")
        mc[1].metric("xBA",      _fmt(xs,"xba"),   delta="lg .248")
        mc[2].metric("xSLG",     _fmt(xs,"xslg"),  delta="lg .411")
        mc[3].metric("Barrel%",  barrel_v,          delta="lg 8.0%")
        mc[4].metric("Exit Velo",ev_v,              delta="lg 88.5 mph")

        if split.get("pa",0) >= 1:
            st.markdown(f"**vs {'LHP' if ph=='L' else 'RHP'} splits**")
            sc = st.columns(5)
            sc[0].metric("AVG", split.get("avg","—"))
            sc[1].metric("SLG", split.get("slg","—"))
            sc[2].metric("OPS", split.get("ops","—"))
            sc[3].metric("HR",  f"{split.get('hr',0)}")
            sc[4].metric("PA",  f"{split.get('pa',0)}")

        st.plotly_chart(advanced_stats_chart(xs, splits, ph),
                        use_container_width=True, config={"displayModeBar":False})

        pstats = p.get("pitcher_stats",{})
        st.markdown(f"**Opposing Pitcher: {p['pitcher']}** ({hand_lbl})")
        pc = st.columns(4)
        def _pf(k, fmt=".2f"):
            v = pstats.get(k)
            return format(v, fmt) if isinstance(v, float) else "—"
        pc[0].metric("ERA",      _pf("era"))
        pc[1].metric("WHIP",     _pf("whip"))
        pc[2].metric("Opp AVG",  _pf("avg_against",".3f"))
        pc[3].metric("HR/9",     _pf("hr_per_9"))
        if pstats.get("l5_avg_against") is not None:
            parts = [f"Opp AVG: **{pstats['l5_avg_against']:.3f}**"]
            if pstats.get("l5_hr_per_9") is not None:
                parts.append(f"HR/9: **{pstats['l5_hr_per_9']:.2f}**")
            if pstats.get("l5_era") is not None:
                parts.append(f"ERA: **{pstats['l5_era']:.2f}**")
            st.caption("Last 5 starts — " + " · ".join(parts))

    with tab_form:
        gl = p.get("game_log",[])
        if gl:
            st.plotly_chart(game_log_chart(gl), use_container_width=True, config={"displayModeBar":False})
            df_gl = pd.DataFrame(gl)
            ab_t=df_gl["ab"].sum(); h_t=df_gl["h"].sum(); hr_t=df_gl["hr"].sum()
            sc2 = st.columns(4)
            sc2[0].metric("H/AB",  f"{h_t}/{ab_t}  ({h_t/max(ab_t,1):.3f})")
            sc2[1].metric("HR/AB", f"{hr_t}/{ab_t}  ({hr_t/max(ab_t,1):.3f})")
            sc2[2].metric("RBI",   int(df_gl["rbi"].sum()))
            sc2[3].metric("K",     int(df_gl["k"].sum()))

            df_gl["H/AB"]  = df_gl.apply(lambda r: f"{r['h']}/{r['ab']}", axis=1)
            df_gl["HR/AB"] = df_gl.apply(lambda r: f"{r['hr']}/{r['ab']}", axis=1)
            show = ["date","opp","H/AB","HR/AB","2b","3b","rbi","bb","k"]
            show = [c for c in show if c in df_gl.columns]
            disp = df_gl[show].copy()
            disp.columns = [c.upper() for c in disp.columns]
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.caption("No game log available.")

    with tab_breakdown:
        st.caption("Bars show how each factor raises (+) or lowers (−) the probability in logit units.")
        hc  = {k:v for k,v in p["hit_contributions"].items() if not k.startswith("_")}
        hrc = {k:v for k,v in p["hr_contributions"].items() if not k.startswith("_")}
        st.plotly_chart(waterfall_chart(hc,  "Hit Probability Drivers"),
                        use_container_width=True, config={"displayModeBar":False})
        st.plotly_chart(waterfall_chart(hrc, "HR Probability Drivers"),
                        use_container_width=True, config={"displayModeBar":False})

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    col_t, col_r = st.columns([7,1])
    with col_t:
        st.title("⚾ MLB Hit & HR Probability Finder")
    with col_r:
        st.write("")
        if st.button("↻ Refresh"):
            st.cache_data.clear(); st.rerun()

    date_str  = today()
    year      = date.today().year

    # ── Game filter bar ────────────────────────────────────────────────────────
    live_games = get_live_scores(date_str)
    selected_ids: set = set()

    if live_games:
        # Build label → game_id map
        game_options: Dict[str, int] = {}
        for g in live_games:
            ha = g["away_abbrev"]; hh = g["home_abbrev"]
            if g["state"] == "Live":
                ih = "▲" if "top" in g.get("inning_half","").lower() else "▼"
                lbl = f"🔴 {ha} {g['away_score']}–{g['home_score']} {hh}  {ih}{g['inning']}"
            elif g["state"] == "Final":
                lbl = f"✓ {ha} {g['away_score']}–{g['home_score']} {hh}  Final"
            else:
                try:
                    gt    = datetime.fromisoformat(g["game_time"].replace("Z","+00:00"))
                    local = gt.astimezone()
                    sub   = local.strftime("%-I:%M %p")
                except:
                    sub = "TBD"
                lbl = f"{ha} @ {hh}  {sub}"
            game_options[lbl] = g["game_id"]

        selected_labels = st.multiselect(
            "Filter by game (leave empty for all)",
            options=list(game_options.keys()),
            placeholder="All games — tap to filter…",
        )
        selected_ids = {game_options[l] for l in selected_labels}

    st.divider()

    # ── Load analysis ─────────────────────────────────────────────────────────
    with st.spinner("Loading lineups · Savant data · Pitcher stats… (first load ~60 s)"):
        data  = run_analysis(date_str)

    projs = data["projections"]
    if not projs:
        st.error("No games or player data found. Check your connection.")
        return

    if selected_ids:
        projs = [p for p in projs if p["game_id"] in selected_ids]

    # ── Model badge ───────────────────────────────────────────────────────────
    model_type = data.get("model", "rule-based")
    minfo      = data.get("model_info") or {}
    if model_type == "ml":
        trained_date = minfo.get("trained_date", "")[:10]
        hit_auc = minfo.get("hit_metrics", {}).get("roc_auc", "—")
        hr_auc  = minfo.get("hr_metrics",  {}).get("roc_auc", "—")
        st.success(
            f"**ML model active** (trained {trained_date}) · "
            f"Hit AUC {hit_auc} · HR AUC {hr_auc}",
            icon="🤖",
        )
    else:
        st.info("Rule-based model active. Run `mlb_data_collector.py` then `mlb_train_model.py` to enable the ML model.", icon="📐")

    # ── Summary metrics ────────────────────────────────────────────────────────
    best_hit = max(projs, key=lambda x: x["p_hit"]) if projs else None
    best_hr  = max(projs, key=lambda x: x["p_hr"])  if projs else None
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Players Analyzed", len(projs))
    m2.metric("Best P(Hit)",  f"{best_hit['p_hit']*100:.1f}%  {best_hit['player']}" if best_hit else "—")
    m3.metric("Best P(HR)",   f"{best_hr['p_hr']*100:.1f}%  {best_hr['player']}"   if best_hr  else "—")
    m4.metric("Games",        len({p["game_id"] for p in projs}))

    st.divider()

    # ── Filters ────────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2,2,2])
    sort_opt    = fc1.selectbox("Sort by", ["P(Hit)","P(HR)","Player Name"])
    search      = fc2.text_input("Search player / team", placeholder="Judge, Yankees…")
    hand_filter = fc3.selectbox("Pitcher hand", ["All","vs LHP","vs RHP"])

    if search:
        q = search.lower()
        projs = [p for p in projs if q in p["player"].lower() or q in p["team"].lower()]
    if hand_filter == "vs LHP": projs = [p for p in projs if p["pitcher_hand"]=="L"]
    if hand_filter == "vs RHP": projs = [p for p in projs if p["pitcher_hand"]=="R"]

    sort_map = {"P(Hit)":"p_hit","P(HR)":"p_hr","Player Name":"player"}
    rev = sort_opt != "Player Name"
    projs = sorted(projs, key=lambda x: x.get(sort_map[sort_opt], 0), reverse=rev)

    if not projs:
        st.info("No players match filters.")
        return

    # ── Player table ───────────────────────────────────────────────────────────
    st.caption("◦ = roster fallback (lineup not yet confirmed)")
    rows = []
    for p in projs:
        xs    = p.get("xstats",{})
        split = p.get("splits",{}).get("vl" if p["pitcher_hand"]=="L" else "vr",{})
        def _xs(k, fmt=".3f"):
            v = _safe_float(xs, k)
            return format(v, fmt) if v is not None else "—"
        def _sp(k, fmt=".3f"):
            v = _safe_float(split, k)
            return format(v, fmt) if v is not None else "—"
        hl = "⬅L" if p["pitcher_hand"]=="L" else "➡R"
        rows.append({
            "Player":      p["player"] + ("" if p.get("confirmed") else " ◦"),
            "Team":        TEAM_ABBREV.get(p["team"], p["team"][:3]),
            "vs Pitcher":  f"{p['pitcher'][:16]} ({hl})",
            "P(Hit)":      f"{p['p_hit']*100:.1f}%",
            "P(HR)":       f"{p['p_hr']*100:.1f}%",
            "BA split":    _sp("avg"),
            "xBA":         _xs("xba"),
            "xSLG":        _xs("xslg"),
            "Venue":       p["venue"][:22],
            "Key Factors": top_factors(p["hit_contributions"], p["hr_contributions"]),
        })

    df_t = pd.DataFrame(rows)
    st.dataframe(
        df_t,
        use_container_width=True,
        hide_index=True,
        height=min(700, 40 + 52 * len(rows)),
        column_config={
            "Key Factors": st.column_config.TextColumn(
                "Key Factors", width="large",
            ),
        },
    )

    # ── Player detail modal ────────────────────────────────────────────────────
    st.divider()
    all_names = [p["player"] for p in projs]
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        sel = st.selectbox("Select a player for detailed breakdown", ["— select —"]+all_names)
    with col_btn:
        st.write("")
        st.write("")
        open_modal = st.button("View Detail", type="primary", disabled=(sel=="— select —"))

    if open_modal and sel != "— select —":
        show_player_modal(sel, data["projections"])

if __name__ == "__main__":
    main()

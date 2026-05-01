#!/usr/bin/env python3
"""
MLB Prop Finder  –  http://localhost:5050
Setup:  pip install flask requests pandas numpy
Run:    ODDS_API_KEY=your_key python mlb_betting.py
"""

import os, math, time, logging, threading
from io import StringIO
from datetime import date
from typing import Optional

import requests, pandas as pd, numpy as np
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
def today() -> str: return date.today().isoformat()
SEASON       = date.today().year
MLB_API      = "https://statsapi.mlb.com/api/v1"
ODDS_API     = "https://api.the-odds-api.com/v4"
SAVANT_BASE  = "https://baseballsavant.mlb.com"
AVG_PA_PER_GAME   = 3.8
LEAGUE_AVG_BA     = 0.248
LEAGUE_AVG_HR_PA  = 0.033

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
    "Petco Park":0.94,"Oracle Park":0.92,"Tropicana Field":0.90,
    "T-Mobile Park":0.93,"Comerica Park":0.94,"PNC Park":0.95,
}

_cache: dict = {}
_cache_lock = threading.Lock()
def cached(key, fn, ttl=1800):
    with _cache_lock:
        e = _cache.get(key)
        if e and (time.time()-e["ts"]) < ttl: return e["data"]
    r = fn()
    with _cache_lock: _cache[key] = {"data":r,"ts":time.time()}
    return r

# ── MLB API ────────────────────────────────────────────────────────────────────
def get_todays_games():
    r = requests.get(f"{MLB_API}/schedule", params={"sportId":1,"date":today(),
        "hydrate":"lineups,probablePitcher(note),team,venue"}, timeout=15)
    r.raise_for_status()
    games = []
    for d in r.json().get("dates",[]):
        for g in d.get("games",[]):
            info = {
                "game_id":      g["gamePk"],
                "game_time":    g.get("gameDate",""),
                "venue":        g.get("venue",{}).get("name","Unknown"),
                "home_team":    g["teams"]["home"]["team"]["name"],
                "away_team":    g["teams"]["away"]["team"]["name"],
                "home_team_id": g["teams"]["home"]["team"]["id"],
                "away_team_id": g["teams"]["away"]["team"]["id"],
                "home_abbrev":  g["teams"]["home"]["team"].get("abbreviation",""),
                "away_abbrev":  g["teams"]["away"]["team"].get("abbreviation",""),
            }
            for side in ("home","away"):
                pp = g["teams"][side].get("probablePitcher",{})
                info[f"{side}_pitcher_id"]   = pp.get("id")
                info[f"{side}_pitcher_name"] = pp.get("fullName","TBD")
            lineups = g.get("lineups",{})
            for side in ("home","away"):
                info[f"{side}_lineup"] = [{"id":p["id"],"name":p.get("fullName",""),"confirmed":True}
                    for p in lineups.get(f"{side}Players",[])]
            games.append(info)
    for game in games:
        for side in ("home","away"):
            if not game[f"{side}_lineup"]:
                roster = get_team_roster(game[f"{side}_team_id"])
                game[f"{side}_lineup"] = [{**p,"confirmed":False} for p in roster]
    return games

def get_team_roster(team_id):
    try:
        r = requests.get(f"{MLB_API}/teams/{team_id}/roster",
            params={"rosterType":"active","season":date.today().year}, timeout=10)
        r.raise_for_status()
        return [{"id":p["person"]["id"],"name":p["person"]["fullName"]}
            for p in r.json().get("roster",[])
            if p.get("position",{}).get("abbreviation","P") not in ("P","SP","RP","CL")]
    except Exception as e:
        log.warning(f"roster {team_id}: {e}"); return []

def normalize_name(name):
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii","ignore").decode().lower().replace(".","").replace("-"," ").replace("'","").strip()

def get_player_splits(player_id):
    try:
        r = requests.get(f"{MLB_API}/people/{player_id}/stats",
            params={"stats":"statSplits","season":SEASON,"group":"hitting","sitCodes":"vl,vr"}, timeout=10)
        r.raise_for_status()
        splits = {}
        for sp in r.json().get("stats",[{}])[0].get("splits",[]):
            code = sp.get("split",{}).get("code","")
            s    = sp.get("stat",{})
            pa   = int(s.get("plateAppearances",0) or 0)
            hr   = int(s.get("homeRuns",0) or 0)
            splits[code] = {
                "avg":  float(s.get("avg",  0) or 0),
                "obp":  float(s.get("obp",  0) or 0),
                "slg":  float(s.get("slg",  0) or 0),
                "ops":  float(s.get("ops",  0) or 0),
                "hr":   hr, "pa": pa,
                "hr_rate": hr/max(pa,1),
            }
        return splits
    except Exception as e:
        log.debug(f"splits {player_id}: {e}"); return {}

def get_player_game_log(player_id, last_n=20):
    try:
        r = requests.get(f"{MLB_API}/people/{player_id}/stats",
            params={"stats":"gameLog","season":SEASON,"group":"hitting"}, timeout=10)
        r.raise_for_status()
        raw = sorted(r.json().get("stats",[{}])[0].get("splits",[]),
            key=lambda x:x.get("date",""), reverse=True)[:last_n]
        return [{"date":g.get("date",""),"opp":g.get("opponent",{}).get("abbreviation",""),
            "ab":g["stat"].get("atBats",0),"h":g["stat"].get("hits",0),
            "hr":g["stat"].get("homeRuns",0),"rbi":g["stat"].get("rbi",0),
            "bb":g["stat"].get("baseOnBalls",0),"k":g["stat"].get("strikeOuts",0)} for g in raw]
    except Exception as e:
        log.debug(f"gamelog {player_id}: {e}"); return []

def get_pitcher_hand(pitcher_id):
    if not pitcher_id: return "R"
    try:
        r = requests.get(f"{MLB_API}/people/{pitcher_id}", timeout=10); r.raise_for_status()
        return r.json().get("people",[{}])[0].get("pitchHand",{}).get("code","R") or "R"
    except: return "R"

def get_pitcher_stats(pitcher_id):
    d = {"era":4.50,"whip":1.30,"hr_per_9":1.20,"avg_against":0.250,"k_per_9":8.5}
    if not pitcher_id: return d
    try:
        r = requests.get(f"{MLB_API}/people/{pitcher_id}/stats",
            params={"stats":"season","season":SEASON,"group":"pitching"}, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats",[{}])[0].get("splits",[])
        if not splits: return d
        s = splits[0].get("stat",{})
        return {
            "era":         float(s.get("era",        4.50) or 4.50),
            "whip":        float(s.get("whip",       1.30) or 1.30),
            "hr_per_9":    float(s.get("homeRunsPer9",1.20) or 1.20),
            "avg_against": float(s.get("avg",        0.250) or 0.250),
            "k_per_9":     float(s.get("strikeoutsPer9",8.5) or 8.5),
        }
    except: return d

def get_player_season_stats(player_id):
    try:
        r = requests.get(f"{MLB_API}/people/{player_id}/stats",
            params={"stats":"season","season":SEASON,"group":"hitting"}, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats",[{}])[0].get("splits",[])
        if not splits: return {}
        s = splits[0].get("stat",{})
        return {"ba":float(s.get("avg",0) or 0),"obp":float(s.get("obp",0) or 0),
                "slg":float(s.get("slg",0) or 0),"ops":float(s.get("ops",0) or 0),
                "hr":int(s.get("homeRuns",0) or 0),"ab":int(s.get("atBats",0) or 0)}
    except Exception as e:
        log.debug(f"season stats {player_id}: {e}"); return {}

def get_savant_xstats():
    try:
        r = requests.get(f"{SAVANT_BASE}/expected_statistics",
            params={"type":"batter","year":SEASON,"position":"","team":"","min":25,"csv":"true"},
            headers={"User-Agent":"Mozilla/5.0"}, timeout=25)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df = df.rename(columns={"player_id":"mlb_id","est_ba":"xba","est_slg":"xslg",
            "est_woba":"xwoba","barrel_batted_rate":"barrel_pct"})
        if "mlb_id" not in df.columns: return pd.DataFrame()
        df["mlb_id"] = df["mlb_id"].astype(str)
        return df.set_index("mlb_id")
    except Exception as e:
        log.warning(f"Savant: {e}"); return pd.DataFrame()

# ── Odds API ───────────────────────────────────────────────────────────────────
PROP_MARKETS = ["batter_hits","batter_home_runs","batter_rbis","batter_total_bases"]
BET_LABELS   = {
    "batter_hits":        "Anytime Hit",
    "batter_home_runs":   "Anytime HR",
    "batter_rbis":        "Anytime RBI",
    "batter_total_bases": "Total Bases 2.5+",
}

def get_mlb_events():
    if not ODDS_API_KEY: return []
    try:
        r = requests.get(f"{ODDS_API}/sports/baseball_mlb/events",
            params={"apiKey":ODDS_API_KEY,"dateFormat":"iso"}, timeout=12)
        if r.status_code == 401: log.error("Odds API: invalid key"); return []
        r.raise_for_status()
        return [e for e in r.json() if e.get("commence_time","")[:10] == today()]
    except: return []

def get_event_props(event_id):
    if not ODDS_API_KEY: return []
    try:
        r = requests.get(f"{ODDS_API}/sports/baseball_mlb/events/{event_id}/odds",
            params={"apiKey":ODDS_API_KEY,"regions":"us","markets":",".join(PROP_MARKETS),
                "oddsFormat":"american","bookmakers":"draftkings,fanduel"}, timeout=15)
        r.raise_for_status()
        return r.json().get("bookmakers",[])
    except: return []

def aggregate_best_odds(bookmakers):
    """
    Returns dict keyed as  "norm_name|market|threshold"
    so Over 0.5 and Over 1.5 hits are tracked separately.
    """
    best = {}
    for book in bookmakers:
        bname = book.get("title","")
        for mkt in book.get("markets",[]):
            mkt_key = mkt.get("key","")
            for oc in mkt.get("outcomes",[]):
                if oc.get("name","").lower() not in ("over","yes"): continue
                threshold = float(oc.get("point", 0.5) or 0.5)
                player    = oc.get("description", oc.get("name",""))
                odds      = int(oc.get("price",0) or 0)
                key = f"{normalize_name(player)}|{mkt_key}|{threshold}"
                if key not in best or odds > best[key]["odds"]:
                    best[key] = {"player":player,"market":mkt_key,
                        "odds":odds,"book":bname,"threshold":threshold}
    return best

def american_to_decimal(o): return (o/100+1) if o>=0 else (100/abs(o)+1)
def implied_prob(o):         return (100/(o+100)) if o>=0 else (abs(o)/(abs(o)+100))
def calc_ev(p, o):           return round((p*(american_to_decimal(o)-1)-(1-p))*100, 2)
def fmt_odds(o):             return f"+{o}" if o>=0 else str(o)

def _poisson_cdf(k, lam):
    res, term = 0.0, math.exp(-lam)
    for i in range(k+1): res += term; term *= lam/(i+1)
    return min(res,1.0)

# ── Stat helpers ───────────────────────────────────────────────────────────────
def _recent_avg(games, n):
    g=games[:n]; ab=sum(x["ab"] for x in g); h=sum(x["h"] for x in g)
    return (h/ab) if ab>=5 else None
def _recent_hr_rate(games, n):
    g=games[:n]; pa=sum(x["ab"]+x.get("bb",0) for x in g); hrs=sum(x["hr"] for x in g)
    return (hrs/pa) if pa>=5 else None
def _ba_from_log(gl, n):
    g=gl[:n]; ab=sum(x["ab"] for x in g); h=sum(x["h"] for x in g)
    return round(h/ab,3) if ab>=1 else 0.0
def _hr_from_log(gl, n): return sum(x["hr"] for x in gl[:n])

# ── Hit probability model ──────────────────────────────────────────────────────
# Root cause of the 92.8% cap problem was the logistic model saturating near
# the 0.50 per-PA hard cap for nearly every player with decent stats.
# New approach: multiplicative model anchored to realistic per-PA rates so that
# batter skill + pitcher quality + park + form each contribute distinct signal.

def _hit_pa_rate(splits, ph, games, venue, pstats, savant=None) -> float:
    """
    Per-PA hit probability used for both P(≥1 hit) and P(≥2 hits).
    Range: [0.12, 0.40]   →   game P(≥1): ~39% – 84%
    """
    split     = splits.get("vl" if ph=="L" else "vr", {})
    split_avg = float(split.get("avg", 0) or 0)
    split_pa  = int(split.get("pa",  0) or 0)

    # Statcast xBA (best forward-looking estimator of true BA)
    xba = None
    if savant:
        try: xba = float(savant.get("xba") or "")
        except: pass

    # ── Hitter skill: blend split BA + xBA weighted by split sample size ──────
    if split_pa >= 100 and split_avg > 0:
        # Large sample: trust the split heavily
        hitter = 0.55 * split_avg + 0.45 * (xba or split_avg)
    elif split_pa >= 30 and split_avg > 0:
        # Moderate sample: blend toward xBA
        w = split_pa / 200          # e.g. 60 PA → w=0.30
        hitter = w * split_avg + (1 - w) * (xba or LEAGUE_AVG_BA)
    else:
        # Small/no split sample: rely on xBA or league average
        hitter = xba or LEAGUE_AVG_BA

    # ── Pitcher quality: BAA relative to league average ───────────────────────
    # Ace (BAA=.200): factor 0.200/0.248 = 0.81  →  reduces hit prob 19%
    # Avg (BAA=.248): factor 1.00              →  no adjustment
    # Weak (BAA=.290): factor 1.17            →  increases hit prob 17%
    pitch_m = pstats.get("avg_against", LEAGUE_AVG_BA) / LEAGUE_AVG_BA
    pitch_m = max(0.70, min(1.35, pitch_m))

    # ── Pitcher strikeout rate: high K% reduces contact opportunities ──────────
    k9 = pstats.get("k_per_9", 8.5)
    # League avg ~8.5 K/9. Scale: 12 K/9 = 0.92x, 5 K/9 = 1.04x
    k_m = max(0.88, min(1.06, 1.0 + (8.5 - k9) * 0.01))

    # ── Park factor ────────────────────────────────────────────────────────────
    park_m = max(0.88, min(1.15, HIT_PARK.get(venue, 1.0)))

    # ── Recent form: L5 and L10 trends (max ±12% combined) ───────────────────
    form_m = 1.0
    avg_l5  = _recent_avg(games, 5)
    avg_l10 = _recent_avg(games, 10)
    if avg_l5  is not None:
        form_m += 0.09 * (avg_l5  - LEAGUE_AVG_BA) / LEAGUE_AVG_BA
    if avg_l10 is not None:
        form_m += 0.05 * (avg_l10 - LEAGUE_AVG_BA) / LEAGUE_AVG_BA
    form_m = max(0.88, min(1.12, form_m))

    pa_rate = hitter * pitch_m * k_m * park_m * form_m
    return max(0.12, min(0.40, pa_rate))


def hit_prob(splits, ph, games, venue, pstats, savant=None) -> float:
    """P(≥1 hit in game).  Spread ~39%–84% depending on matchup."""
    p = _hit_pa_rate(splits, ph, games, venue, pstats, savant)
    return round(1.0 - (1.0 - p) ** AVG_PA_PER_GAME, 4)


def two_hit_prob(splits, ph, games, venue, pstats, savant=None) -> float:
    """
    P(≥2 hits in game) — correct probability for the Over 1.5 hits bet.
    Uses Poisson with λ = pa_rate × avg_PA_per_game.
    """
    p   = _hit_pa_rate(splits, ph, games, venue, pstats, savant)
    lam = p * AVG_PA_PER_GAME        # expected hits per game
    p0  = math.exp(-lam)             # P(0 hits)
    p1  = lam * math.exp(-lam)       # P(exactly 1 hit)
    return round(max(0.0, 1.0 - p0 - p1), 4)


# ── HR probability (logistic — already shows good spread) ─────────────────────
def hr_prob(splits, ph, games, venue, pstats, savant=None) -> float:
    split = splits.get("vl" if ph=="L" else "vr", {})

    barrel_pct, xslg = 8.0, 0.411
    if savant:
        try: barrel_pct = float(savant.get("barrel") or "") or barrel_pct
        except: pass
        try: xslg = float(savant.get("xslg") or "") or xslg
        except: pass

    split_pa  = int(split.get("pa",0) or 0)
    split_hr  = int(split.get("hr",0) or 0)
    split_rate = (split_hr/split_pa) if split_pa>=30 else LEAGUE_AVG_HR_PA

    def logit(p): p=max(1e-6,min(1-1e-6,p)); return math.log(p/(1-p))
    def ladj(r):  return max(-1.5,min(1.5,math.log(max(r,1e-4))))

    z  = logit(LEAGUE_AVG_HR_PA)                          # ≈ -3.37
    z += 1.8 * ladj(barrel_pct / 8.0)                    # barrel% (strongest)
    z += 1.0 * ladj(xslg / 0.411)                        # xSLG
    if split_pa >= 30:
        z += 1.4 * ladj(split_rate / LEAGUE_AVG_HR_PA)   # platoon split HR rate
    hr_l5  = _recent_hr_rate(games,5)
    hr_l10 = _recent_hr_rate(games,10)
    if hr_l5  is not None: z += 0.9*ladj((hr_l5+LEAGUE_AVG_HR_PA)/(2*LEAGUE_AVG_HR_PA))
    if hr_l10 is not None: z += 0.6*ladj((hr_l10+LEAGUE_AVG_HR_PA)/(2*LEAGUE_AVG_HR_PA))
    z += 0.6 * ladj(HR_PARK.get(venue,1.0))
    z += 0.8 * ladj(pstats.get("hr_per_9",1.20)/1.20)

    hr_rate = max(0.003, min(0.13, 1.0/(1.0+math.exp(-z))))
    return round(1.0-(1.0-hr_rate)**AVG_PA_PER_GAME, 4)


def total_bases_prob(splits, ph, games, venue, pstats, savant=None, threshold=2.5) -> float:
    split = splits.get("vl" if ph=="L" else "vr", {})
    xslg  = 0.411
    if savant:
        try: xslg = float(savant.get("xslg") or "") or xslg
        except: pass
    slg = float(split.get("slg") or xslg)
    exp_tb = max(0.1,(0.55*xslg+0.45*slg)*AVG_PA_PER_GAME
        *HR_PARK.get(venue,1.0)*(pstats.get("avg_against",LEAGUE_AVG_BA)/LEAGUE_AVG_BA))
    return round(max(0.01,min(0.99,1.0-_poisson_cdf(int(threshold-0.001),exp_tb))),4)


def rbi_prob(splits, ph, games, venue, pstats, savant=None) -> float:
    hp = hit_prob(splits, ph, games, venue, pstats, savant)
    return round(1.0-(1.0-(hp/AVG_PA_PER_GAME)*0.28)**AVG_PA_PER_GAME, 4)


# ── Build full projection object ───────────────────────────────────────────────
def _build_proj(pid, name, game, side, phand, pstats, splits, gl, xstats, sstats,
                confirmed=False, odds_map=None) -> dict:
    opp   = "away" if side=="home" else "home"
    venue = game["venue"]
    vl, vr = splits.get("vl",{}), splits.get("vr",{})

    p_hit   = hit_prob       (splits, phand, gl, venue, pstats, xstats)
    p_2hits = two_hit_prob   (splits, phand, gl, venue, pstats, xstats)
    p_hr    = hr_prob        (splits, phand, gl, venue, pstats, xstats)
    p_tb    = total_bases_prob(splits, phand, gl, venue, pstats, xstats)
    p_rbi   = rbi_prob       (splits, phand, gl, venue, pstats, xstats)

    proj = {
        "player":       name,
        "player_id":    pid,
        "team":         game[f"{side}_team"],
        "team_id":      game[f"{side}_team_id"],
        "game_id":      game["game_id"],
        "venue":        venue,
        "pitcher":      game.get(f"{opp}_pitcher_name","TBD"),
        "pitcher_hand": phand,
        "confirmed":    confirmed,
        "p_hit":        p_hit,
        "p_2hits":      p_2hits,
        "p_hr":         p_hr,
        "p_tb":         p_tb,
        "p_rbi":        p_rbi,
        "xstats":       xstats,
        "split":        splits.get("vl" if phand=="L" else "vr", {}),
        "split_vl":     vl,
        "split_vr":     vr,
        "game_log":     gl,
        "ba_vs_left":   round(float(vl.get("avg",0) or 0),3),
        "ba_vs_right":  round(float(vr.get("avg",0) or 0),3),
        "ba_last5":     _ba_from_log(gl,5),
        "ba_last10":    _ba_from_log(gl,10),
        "ba_last20":    _ba_from_log(gl,20),
        "hr_last5":     _hr_from_log(gl,5),
        "hr_last10":    _hr_from_log(gl,10),
        "hr_last20":    _hr_from_log(gl,20),
        "season_ba":    round(float(sstats.get("ba",0)),3),
        "season_obp":   round(float(sstats.get("obp",0)),3),
        "season_slg":   round(float(sstats.get("slg",0)),3),
        "season_ops":   round(float(sstats.get("ops",0)),3),
        "season_hr":    int(sstats.get("hr",0)),
        # DK odds & EV (populated below when odds_map provided)
        "dk_hit_odds":   None,
        "ev_hit":        None,
        "dk_2hit_odds":  None,
        "ev_2hit":       None,
        "dk_hr_odds":    None,
        "ev_hr":         None,
    }

    if odds_map:
        nname = normalize_name(name)
        # Over 0.5 hits (anytime hit)
        od05 = odds_map.get(f"{nname}|batter_hits|0.5")
        if od05:
            proj["dk_hit_odds"] = od05["odds"]
            proj["ev_hit"]      = calc_ev(p_hit, od05["odds"])
        # Over 1.5 hits (2+ hits) — use p_2hits for correct EV
        od15 = odds_map.get(f"{nname}|batter_hits|1.5")
        if od15:
            proj["dk_2hit_odds"] = od15["odds"]
            proj["ev_2hit"]      = calc_ev(p_2hits, od15["odds"])
        # Anytime HR
        odhr = odds_map.get(f"{nname}|batter_home_runs|0.5")
        if odhr:
            proj["dk_hr_odds"] = odhr["odds"]
            proj["ev_hr"]      = calc_ev(p_hr, odhr["odds"])

    return proj


# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>MLB Prop Finder</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --blue:#1565C0;--blue2:#1976D2;--blue3:#0D47A1;
  --blue-bg:#E3F2FD;--blue-light:#BBDEFB;
  --yellow:#FFC107;
  --white:#FFFFFF;--page-bg:#EEF3F9;
  --text:#1A1A2E;--text2:#455A64;--text3:#90A4AE;
  --border:#CFE0EE;
  --green:#1B5E20;--green2:#2E7D32;--green-bg:#E8F5E9;
  --red:#B71C1C;--red2:#C62828;--red-bg:#FFEBEE;
  --orange:#E65100;--orange2:#BF360C;
}
body{background:var(--page-bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px}

/* NAV */
.site-header{background:linear-gradient(135deg,var(--blue3),var(--blue),var(--blue2));
  padding:10px 24px;display:flex;align-items:center;gap:16px;
  border-bottom:3px solid var(--yellow);position:sticky;top:0;z-index:100}
.nav-pill{display:flex;background:rgba(255,255,255,.15);border-radius:30px;padding:4px;gap:2px}
.nav-btn{padding:6px 18px;border-radius:22px;border:none;background:transparent;
  color:rgba(255,255,255,.8);font-size:.75rem;font-weight:700;letter-spacing:.06em;cursor:pointer;transition:all .15s}
.nav-btn:hover{background:rgba(255,255,255,.2);color:#fff}
.nav-btn.active{background:rgba(255,255,255,.28);color:#fff}
.betfinder-label{display:flex;align-items:center;gap:8px;font-size:.85rem;font-weight:800;
  letter-spacing:.06em;color:var(--yellow);white-space:nowrap}
.nav-search{margin-left:auto;background:rgba(255,255,255,.12);border:1.5px solid rgba(255,255,255,.3);
  border-radius:8px;padding:7px 14px;display:flex;align-items:center;gap:8px;min-width:200px}
.nav-search input{background:none;border:none;outline:none;color:#fff;font-size:.8rem;
  font-weight:600;letter-spacing:.04em;width:100%}
.nav-search input::placeholder{color:rgba(255,255,255,.5);font-weight:400}

/* SCORES */
.scores-bar{background:var(--white);border-bottom:1px solid var(--border);
  padding:10px 24px;display:flex;align-items:center;gap:8px;
  overflow-x:auto;scrollbar-width:none}
.scores-bar::-webkit-scrollbar{display:none}
.all-btn{padding:6px 16px;border-radius:6px;background:var(--blue);border:none;
  color:#fff;font-size:.74rem;font-weight:700;letter-spacing:.05em;cursor:pointer;flex-shrink:0;transition:all .15s}
.all-btn.inactive{background:transparent;border:1.5px solid var(--blue);color:var(--blue)}
.all-btn.inactive:hover{background:var(--blue-bg)}
.score-box{flex-shrink:0;background:var(--white);border:1.5px solid var(--blue);
  border-radius:8px;padding:6px 14px;cursor:pointer;transition:all .15s;min-width:96px}
.score-box:hover{background:var(--blue-bg)}
.score-box.active{background:var(--blue)}
.score-box.active .score-line,.score-box.active .score-st{color:#fff!important}
.score-st{font-size:.62rem;font-weight:700;letter-spacing:.05em;color:var(--text3);
  text-transform:uppercase;margin-bottom:3px}
.score-st.live{color:#E53935}
.score-line{font-size:.8rem;font-weight:700;color:var(--text);line-height:1.55}

/* PAGE */
.page-body{padding:18px 24px;max-width:1500px;margin:0 auto}
.section-card{background:var(--white);border:1px solid var(--border);
  border-left:5px solid var(--blue);border-radius:0 8px 8px 0;overflow:hidden}
.section-title{background:var(--blue-bg);padding:10px 18px;font-size:.74rem;font-weight:700;
  letter-spacing:.06em;color:var(--blue3);text-transform:uppercase;border-bottom:1px solid var(--blue-light)}

/* TABLE */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:1000px}
thead th{background:var(--blue3);color:rgba(255,255,255,.85);padding:9px 10px;
  font-size:.66rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
  white-space:nowrap;position:sticky;top:0;z-index:5;cursor:default}
thead th.sortable{cursor:pointer}
thead th.sortable:hover{background:var(--blue2)}
thead th.sorted{background:var(--blue2);color:var(--yellow)}
thead th.sorted::after{content:' ↓';font-size:.6rem}
thead th.sorted.asc::after{content:' ↑'}
tbody tr{border-bottom:1px solid #EEF3F9;cursor:pointer;transition:background .08s}
tbody tr:hover td{background:#F0F7FF}
tbody td{padding:8px 10px;vertical-align:middle;white-space:nowrap}
.cell-player{font-weight:700;color:var(--blue);font-size:.87rem;text-decoration:underline;text-underline-offset:2px}
.cell-hand{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.63rem;font-weight:700;margin-top:2px}
.hand-L{background:#E1F5FE;color:#0277BD}
.hand-R{background:#FFF3E0;color:#E65100}
.prob-hit{font-family:monospace;font-weight:700;font-size:.88rem;color:var(--green2)}
.prob-hr {font-family:monospace;font-weight:700;font-size:.88rem;color:var(--orange)}
.prob-2h {font-family:monospace;font-weight:700;font-size:.88rem;color:#6A1B9A}
.stat-n{font-family:monospace;font-size:.82rem;color:var(--text)}
.stat-z{color:var(--text3)}
.ba-hot{color:var(--green2)!important;font-weight:700}
.ba-cold{color:var(--red2)!important}
.ev-pos{display:inline-block;background:var(--green-bg);color:var(--green);
  font-weight:700;font-family:monospace;font-size:.78rem;padding:2px 6px;border-radius:4px}
.ev-neg{display:inline-block;background:var(--red-bg);color:var(--red2);
  font-family:monospace;font-size:.78rem;padding:2px 6px;border-radius:4px}
.odds-badge{font-family:monospace;font-size:.75rem;color:var(--text2);margin-bottom:2px}
.empty-cell{text-align:center;padding:40px!important;color:var(--text3)}

/* LOAD/ERR */
.load-area{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px;gap:12px}
.spinner{width:34px;height:34px;border:3px solid var(--border);border-top-color:var(--blue);
  border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.err-box{background:#FFEBEE;border:1px solid #FFCDD2;border-radius:8px;padding:16px 20px;text-align:center;margin:20px 0}

/* MODAL */
#modal-backdrop{position:fixed;inset:0;background:rgba(13,71,161,.55);backdrop-filter:blur(3px);
  z-index:200;display:none;align-items:flex-start;justify-content:center;padding:28px 16px;overflow-y:auto}
#modal-backdrop.open{display:flex}
#player-modal{background:var(--white);border:1px solid var(--border);border-radius:12px;
  width:100%;max-width:820px;box-shadow:0 20px 60px rgba(13,71,161,.25);overflow:hidden;flex-shrink:0}
.modal-hdr{background:linear-gradient(135deg,var(--blue3),var(--blue2));padding:14px 20px;
  display:flex;align-items:center;justify-content:space-between}
.modal-title{color:#fff;font-size:1.05rem;font-weight:800}
.modal-meta{color:rgba(255,255,255,.75);font-size:.78rem;margin-top:2px}
.modal-close{background:rgba(255,255,255,.15);border:none;color:#fff;width:30px;height:30px;
  border-radius:50%;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s}
.modal-close:hover{background:rgba(255,255,255,.3)}
.modal-probs{display:flex;border-bottom:1px solid var(--border)}
.modal-prob-cell{flex:1;border-right:1px solid var(--border);padding:10px 8px;text-align:center}
.modal-prob-cell:last-child{border-right:none}
.modal-prob-lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text3);margin-bottom:4px}
.modal-prob-val{font-size:1.15rem;font-weight:800;font-family:monospace}
.modal-stats-row{display:flex;border-bottom:1px solid var(--border)}
.modal-photo-box{width:130px;min-width:130px;background:var(--blue-bg);border-right:1px solid var(--border);
  display:flex;flex-direction:column;align-items:center;justify-content:center;padding:14px 8px;gap:7px}
.modal-headshot{width:74px;height:74px;border-radius:50%;border:2px solid var(--blue-light);object-fit:cover;background:var(--border)}
.modal-team-logo{width:38px;height:38px;object-fit:contain}
.modal-stat-box{flex:1;border-right:1px solid var(--border);display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:14px 8px;text-align:center}
.modal-stat-box:last-child{border-right:none}
.modal-stat-lbl{font-size:.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--text2);margin-bottom:5px}
.modal-stat-val{font-size:1.45rem;font-weight:800;font-family:monospace;color:var(--blue)}
.modal-chart-box{padding:14px 20px;border-bottom:1px solid var(--border)}
.modal-chart-ttl{font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--text2);margin-bottom:8px}
.modal-controls{display:flex;gap:20px;padding:14px 20px;flex-wrap:wrap;align-items:flex-start;background:#FAFCFF}
.ctrl-grp{display:flex;flex-direction:column;gap:6px;flex:1;min-width:200px}
.ctrl-lbl{font-size:.66rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--text3)}
.hand-toggle{display:flex;border:1.5px solid var(--blue-light);border-radius:8px;overflow:hidden}
.hand-btn{flex:1;padding:7px 8px;border:none;background:var(--white);color:var(--text2);
  font-size:.74rem;font-weight:700;cursor:pointer;transition:all .15s}
.hand-btn:not(:last-child){border-right:1px solid var(--blue-light)}
.hand-btn.active{background:var(--blue);color:#fff}
.hand-btn:hover:not(.active){background:var(--blue-bg)}
.period-wrap{display:flex;flex-wrap:wrap;gap:5px}
.period-btn{padding:5px 11px;border-radius:30px;border:1.5px solid var(--blue-light);
  background:var(--white);color:var(--text2);font-size:.72rem;font-weight:600;cursor:pointer;transition:all .15s}
.period-btn.active{background:var(--yellow);border-color:var(--yellow);color:var(--text);font-weight:700}
.period-btn:hover:not(.active){background:var(--blue-bg)}
@media(max-width:600px){
  .modal-stats-row{flex-wrap:wrap}
  .modal-photo-box{width:100%;min-width:unset;flex-direction:row;border-right:none;border-bottom:1px solid var(--border)}
  .modal-stat-box{min-width:calc(50% - 1px)}
}
</style>
</head>
<body>

<header class="site-header">
  <div class="nav-pill">
    <button class="nav-btn active">HOME</button>
    <button class="nav-btn">NEWS</button>
    <button class="nav-btn">CONTACT</button>
  </div>
  <div class="betfinder-label"><span>⚾</span> BET FINDER</div>
  <div class="nav-search">
    <svg width="13" height="13" fill="none" stroke="white" stroke-width="2" viewBox="0 0 20 20"><circle cx="9" cy="9" r="7"/><line x1="15" y1="15" x2="19" y2="19"/></svg>
    <input type="text" id="navSearch" placeholder="SEARCH PLAYER NAME" oninput="applyFilters()"/>
  </div>
</header>

<div class="scores-bar" id="scoresBar">
  <button class="all-btn" id="allBtn" onclick="filterByGame(null)">ALL GAMES</button>
  <span style="color:var(--text3);font-size:.8rem">Loading scores…</span>
</div>

<div class="page-body">
  <div id="loadDiv" class="load-area">
    <div class="spinner"></div>
    <p style="color:var(--text2);font-size:.9rem">Fetching lineups &amp; calculating probabilities…</p>
    <p style="color:var(--text3);font-size:.78rem">First load takes 30–60 s</p>
  </div>
  <div id="errDiv" style="display:none" class="err-box">
    <div style="color:var(--red2);font-weight:700;margin-bottom:6px" id="errMsg"></div>
    <div style="color:var(--text2);font-size:.85rem">Check network and try refreshing.</div>
  </div>

  <div id="tableSection" style="display:none" class="section-card">
    <div class="section-title" id="tableTitle">TABLE OF PLAYERS — DEFAULT SORT: PROBABILITY OF HIT (DESCENDING)</div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th style="text-align:left">PLAYER NAME</th>
            <th style="text-align:left">PLAYER TEAM</th>
            <th style="text-align:left">VS PITCHER</th>
            <th class="sortable sorted" data-col="p_hit"       onclick="sortBy('p_hit')"       style="text-align:right">P(HIT)</th>
            <th class="sortable"        data-col="p_2hits"     onclick="sortBy('p_2hits')"     style="text-align:right">P(2H)</th>
            <th class="sortable"        data-col="p_hr"        onclick="sortBy('p_hr')"        style="text-align:right">P(HR)</th>
            <th class="sortable"        data-col="season_ba"   onclick="sortBy('season_ba')"   style="text-align:right">BA THIS YEAR</th>
            <th class="sortable"        data-col="ba_dyn"      onclick="sortBy('ba_dyn')"      style="text-align:right" id="ba-dyn-hdr">BA LAST 10</th>
            <th class="sortable"        data-col="season_obp"  onclick="sortBy('season_obp')"  style="text-align:right">OBP</th>
            <th class="sortable"        data-col="season_slg"  onclick="sortBy('season_slg')"  style="text-align:right">SLG</th>
            <th class="sortable"        data-col="season_hr"   onclick="sortBy('season_hr')"   style="text-align:right">HR THIS YEAR</th>
            <th class="sortable"        data-col="hr_dyn"      onclick="sortBy('hr_dyn')"      style="text-align:right" id="hr-dyn-hdr">HR LAST 10</th>
            <th class="sortable"        data-col="ba_vs_left"  onclick="sortBy('ba_vs_left')"  style="text-align:right">BA VS LEFT</th>
            <th class="sortable"        data-col="ba_vs_right" onclick="sortBy('ba_vs_right')" style="text-align:right">BA VS RIGHT</th>
            <th id="hit-ev-hdr"         style="text-align:center">DK HIT (O0.5)</th>
            <th id="2h-ev-hdr"          style="text-align:center">DK 2H (O1.5)</th>
            <th id="hr-ev-hdr"          style="text-align:center">DK HR</th>
          </tr>
        </thead>
        <tbody id="playersBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- MODAL -->
<div id="modal-backdrop" onclick="handleBackdrop(event)">
<div id="player-modal" onclick="event.stopPropagation()">
  <div class="modal-hdr">
    <div><div class="modal-title" id="modal-name"></div><div class="modal-meta" id="modal-meta"></div></div>
    <button class="modal-close" onclick="closeModal()">✕</button>
  </div>
  <div class="modal-probs">
    <div class="modal-prob-cell"><div class="modal-prob-lbl">P(Hit)</div><div class="modal-prob-val" style="color:var(--green2)" id="m-phit">—</div></div>
    <div class="modal-prob-cell"><div class="modal-prob-lbl">P(2+ Hits)</div><div class="modal-prob-val" style="color:#6A1B9A" id="m-p2h">—</div></div>
    <div class="modal-prob-cell"><div class="modal-prob-lbl">P(HR)</div><div class="modal-prob-val" style="color:var(--orange)" id="m-phr">—</div></div>
    <div class="modal-prob-cell"><div class="modal-prob-lbl">P(TB 2.5+)</div><div class="modal-prob-val" style="color:#0277BD" id="m-ptb">—</div></div>
  </div>
  <div class="modal-stats-row">
    <div class="modal-photo-box">
      <img id="modal-headshot" class="modal-headshot" src=""
        onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2274%22 height=%2274%22><circle cx=%2237%22 cy=%2237%22 r=%2237%22 fill=%22%23BBDEFB%22/><text x=%2237%22 y=%2250%22 font-size=%2232%22 text-anchor=%22middle%22>⚾</text></svg>'"/>
      <img id="modal-team-logo" class="modal-team-logo" src="" onerror="this.style.display='none'" alt=""/>
    </div>
    <div class="modal-stat-box"><div class="modal-stat-lbl">BA</div><div class="modal-stat-val" id="m-ba">—</div></div>
    <div class="modal-stat-box"><div class="modal-stat-lbl">OBP</div><div class="modal-stat-val" id="m-obp">—</div></div>
    <div class="modal-stat-box"><div class="modal-stat-lbl">SLG</div><div class="modal-stat-val" id="m-slg">—</div></div>
    <div class="modal-stat-box"><div class="modal-stat-lbl">OPS</div><div class="modal-stat-val" id="m-ops">—</div></div>
  </div>
  <div class="modal-chart-box">
    <div class="modal-chart-ttl" id="chart-ttl">HITS &amp; HR — LAST 10 GAMES</div>
    <canvas id="modal-chart" height="150"></canvas>
  </div>
  <div class="modal-controls">
    <div class="ctrl-grp">
      <div class="ctrl-lbl">Stat Split</div>
      <div class="hand-toggle">
        <button class="hand-btn" data-hand="L" onclick="setModalHand('L')">← VS LHP</button>
        <button class="hand-btn active" data-hand="" onclick="setModalHand('')">ALL</button>
        <button class="hand-btn" data-hand="R" onclick="setModalHand('R')">VS RHP →</button>
      </div>
    </div>
    <div class="ctrl-grp">
      <div class="ctrl-lbl">Histogram Period</div>
      <div class="period-wrap">
        <button class="period-btn" data-n="5"      onclick="setModalPeriod(5)">LAST 5</button>
        <button class="period-btn active" data-n="10" onclick="setModalPeriod(10)">LAST 10</button>
        <button class="period-btn" data-n="20"     onclick="setModalPeriod(20)">LAST 20</button>
        <button class="period-btn" data-n="season" onclick="setModalPeriod('season')">THIS SEASON</button>
        <button class="period-btn" data-n="career" onclick="setModalPeriod('career')">CAREER</button>
      </div>
    </div>
  </div>
</div>
</div>

<script>
let allProj = [], activeGame = null;
let sortCol = 'p_hit', sortDir = 1;   // 1 = descending (highest first)
let modalPlayer = null, modalHand = '', modalPeriod = 10, modalChart = null;

const fmt3 = v => v > 0 ? v.toFixed(3).replace(/^0/,'') : '—';
const pct  = v => v ? (v*100).toFixed(1)+'%' : '—';
const fmtOdds = o => o >= 0 ? '+'+o : ''+o;
const evHtml = (ev, odds) => {
  if (ev === null || ev === undefined) return '<span class="stat-z">—</span>';
  const cls = ev >= 0 ? 'ev-pos' : 'ev-neg';
  const sign = ev >= 0 ? '+' : '';
  return `<div class="odds-badge">${fmtOdds(odds)}</div><span class="${cls}">${sign}$${Math.abs(ev).toFixed(2)}</span>`;
};

// ── Scores ─────────────────────────────────────────────────────────────────
async function loadScores() {
  try { renderScores((await (await fetch('/api/scores')).json()).scores || []); } catch(e){}
}
function renderScores(scores) {
  const allActive = activeGame === null;
  let html = `<button class="all-btn ${allActive?'':'inactive'}" onclick="filterByGame(null)">ALL GAMES</button>`;
  scores.forEach(s => {
    const live=s.status==='Live', fin=s.status==='Final';
    let st;
    if (live)      st=`<div class="score-st live">● LIVE ${s.inning}${s.half==='Top'?'▲':'▼'}</div>`;
    else if (fin)  st=`<div class="score-st">FINAL</div>`;
    else {
      const t = s.game_time ? new Date(s.game_time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : 'TBD';
      st = `<div class="score-st">${t} ET</div>`;
    }
    const act = activeGame===s.game_id;
    const lines = (live||fin)
      ? `<div class="score-line">${s.away_score}  ${s.away_team}</div><div class="score-line">${s.home_score}  ${s.home_team}</div>`
      : `<div class="score-line">${s.away_team}</div><div class="score-line">${s.home_team}</div>`;
    html += `<div class="score-box${act?' active':''}" onclick="filterByGame(${s.game_id})">${st}${lines}</div>`;
  });
  document.getElementById('scoresBar').innerHTML = html;
}
function filterByGame(id) {
  activeGame = id;
  loadScores();
  applyFilters();
  document.getElementById('tableTitle').textContent = id===null
    ? 'TABLE OF PLAYERS — DEFAULT SORT: PROBABILITY OF HIT (DESCENDING)'
    : 'TABLE OF PLAYERS — FILTERED: SELECTED GAME';
}

// ── Sort ───────────────────────────────────────────────────────────────────
function sortBy(col) {
  if (sortCol===col) sortDir*=-1; else {sortCol=col; sortDir=1;}
  document.querySelectorAll('thead th.sorted').forEach(t=>{t.classList.remove('sorted');t.classList.remove('asc')});
  const th = document.querySelector(`thead th[data-col="${col}"]`);
  if (th) { th.classList.add('sorted'); if (sortDir===-1) th.classList.add('asc'); }
  applyFilters();
}

// ── Dynamic period columns ─────────────────────────────────────────────────
let tablePeriod = 10;
const getDynBA = p => typeof tablePeriod==='number' ? (p[`ba_last${tablePeriod}`]||0) : p.season_ba||0;
const getDynHR = p => typeof tablePeriod==='number' ? (p[`hr_last${tablePeriod}`]||0) : p.season_hr||0;

function applyFilters() {
  const q = (document.getElementById('navSearch').value||'').toLowerCase();
  let out = allProj.filter(p => {
    if (q && !p.player.toLowerCase().includes(q) && !p.team.toLowerCase().includes(q)) return false;
    if (activeGame!==null && p.game_id!==activeGame) return false;
    return true;
  });
  out.sort((a,b) => {
    let av = sortCol==='ba_dyn' ? getDynBA(a) : sortCol==='hr_dyn' ? getDynHR(a) : (a[sortCol]??0);
    let bv = sortCol==='ba_dyn' ? getDynBA(b) : sortCol==='hr_dyn' ? getDynHR(b) : (b[sortCol]??0);
    return sortDir * ((bv-av)||0);   // sortDir=1 → descending (b-a)
  });
  renderTable(out);
}

function renderTable(projs) {
  const tbody = document.getElementById('playersBody');
  const hasOdds = projs.some(p => p.dk_hit_odds!==null || p.dk_2hit_odds!==null);
  if (!projs.length) {
    tbody.innerHTML=`<tr><td colspan="17" class="empty-cell">No players match filters. Click ALL GAMES to reset.</td></tr>`;
    return;
  }
  tbody.innerHTML = projs.map(p => {
    const safe = n => n.replace(/'/g,"\\'");
    const baD  = getDynBA(p), hrD = getDynHR(p);
    const baC  = baD>=.300?'ba-hot':baD>.250?'':baD>0?'ba-cold':'stat-z';
    const sBaC = (p.season_ba||0)>=.300?'ba-hot':(p.season_ba||0)>.250?'':p.season_ba>0?'ba-cold':'stat-z';
    const phitC = p.p_hit>=.72?'var(--green2)':p.p_hit>=.60?'#558B2F':'var(--orange)';
    return `<tr onclick="openModal('${safe(p.player)}')">
      <td><div class="cell-player">${p.player}${p.confirmed?'':' <span style="color:var(--text3);font-weight:400;font-size:.7rem">(roster)</span>'}</div></td>
      <td><span style="font-size:.8rem;color:var(--text2)">${p.team}</span></td>
      <td>
        <div style="font-size:.8rem">${p.pitcher}</div>
        <span class="cell-hand hand-${p.pitcher_hand}">${p.pitcher_hand==='L'?'LHP':'RHP'}</span>
      </td>
      <td style="text-align:right"><span style="color:${phitC}" class="prob-hit">${pct(p.p_hit)}</span></td>
      <td style="text-align:right"><span class="prob-2h">${pct(p.p_2hits)}</span></td>
      <td style="text-align:right"><span class="prob-hr">${pct(p.p_hr)}</span></td>
      <td style="text-align:right"><span class="stat-n ${sBaC}">${p.season_ba>0?fmt3(p.season_ba):'—'}</span></td>
      <td style="text-align:right"><span class="stat-n ${baC}">${baD>0?fmt3(baD):'—'}</span></td>
      <td style="text-align:right"><span class="stat-n">${p.season_obp>0?fmt3(p.season_obp):'—'}</span></td>
      <td style="text-align:right"><span class="stat-n">${p.season_slg>0?fmt3(p.season_slg):'—'}</span></td>
      <td style="text-align:right"><span class="stat-n">${p.season_hr>0?p.season_hr:'—'}</span></td>
      <td style="text-align:right"><span class="stat-n">${hrD>0?hrD:'—'}</span></td>
      <td style="text-align:right"><span class="stat-n" style="color:#0277BD">${p.ba_vs_left>0?fmt3(p.ba_vs_left):'—'}</span></td>
      <td style="text-align:right"><span class="stat-n" style="color:var(--orange)">${p.ba_vs_right>0?fmt3(p.ba_vs_right):'—'}</span></td>
      <td style="text-align:center">${evHtml(p.ev_hit, p.dk_hit_odds)}</td>
      <td style="text-align:center">${evHtml(p.ev_2hit, p.dk_2hit_odds)}</td>
      <td style="text-align:center">${evHtml(p.ev_hr, p.dk_hr_odds)}</td>
    </tr>`;
  }).join('');
}

// ── Modal ──────────────────────────────────────────────────────────────────
function handleBackdrop(e) { if(e.target===document.getElementById('modal-backdrop')) closeModal(); }
function closeModal() {
  document.getElementById('modal-backdrop').classList.remove('open');
  if (modalChart) { modalChart.destroy(); modalChart=null; }
}
function openModal(name) {
  modalPlayer = allProj.find(p=>p.player===name);
  if (!modalPlayer) return;
  modalHand=''; modalPeriod=10;
  document.querySelectorAll('.hand-btn').forEach(b=>b.classList.toggle('active',b.dataset.hand===''));
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.toggle('active',b.dataset.n==='10'));
  populateModal();
  document.getElementById('modal-backdrop').classList.add('open');
  document.getElementById('player-modal').scrollTop=0;
}
function populateModal() {
  const p = modalPlayer;
  document.getElementById('modal-name').textContent = p.player;
  document.getElementById('modal-meta').textContent = `${p.team} · vs ${p.pitcher} (${p.pitcher_hand==='L'?'LHP':'RHP'}) · ${p.venue}`;
  document.getElementById('modal-headshot').src =
    `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${p.player_id}/headshot/67/current`;
  const logo=document.getElementById('modal-team-logo');
  if(p.team_id){logo.src=`https://www.mlbstatic.com/team-logos/${p.team_id}.svg`;logo.style.display='';}
  else logo.style.display='none';
  document.getElementById('m-phit').textContent = pct(p.p_hit);
  document.getElementById('m-p2h').textContent  = pct(p.p_2hits);
  document.getElementById('m-phr').textContent  = pct(p.p_hr);
  document.getElementById('m-ptb').textContent  = pct(p.p_tb);
  updateModalStats();
  drawHistogram();
}
function updateModalStats() {
  const p=modalPlayer;
  let ba,obp,slg,ops;
  if (modalHand==='L'){const v=p.split_vl||{};ba=v.avg||0;obp=v.obp||0;slg=v.slg||0;ops=v.ops||0;}
  else if (modalHand==='R'){const v=p.split_vr||{};ba=v.avg||0;obp=v.obp||0;slg=v.slg||0;ops=v.ops||0;}
  else {ba=p.season_ba||0;obp=p.season_obp||0;slg=p.season_slg||0;ops=p.season_ops||0;}
  document.getElementById('m-ba').textContent  = ba>0?fmt3(ba):'—';
  document.getElementById('m-obp').textContent = obp>0?fmt3(obp):'—';
  document.getElementById('m-slg').textContent = slg>0?fmt3(slg):'—';
  document.getElementById('m-ops').textContent = ops>0?fmt3(ops):'—';
}
function drawHistogram() {
  const p=modalPlayer;
  const n=typeof modalPeriod==='number'?modalPeriod:10;
  const gl=(p.game_log||[]).slice(0,n).reverse();
  document.getElementById('chart-ttl').textContent=`HITS & HR — LAST ${typeof modalPeriod==='number'?modalPeriod:'SEASON'} GAMES`;
  if(modalChart){modalChart.destroy();modalChart=null;}
  modalChart=new Chart(document.getElementById('modal-chart'),{type:'bar',
    data:{labels:gl.map(g=>g.date?g.date.slice(5):'?'),
      datasets:[
        {label:'Hits',data:gl.map(g=>g.h),backgroundColor:'rgba(21,101,192,.75)',borderRadius:4,borderSkipped:false},
        {label:'HR',  data:gl.map(g=>g.hr),backgroundColor:'rgba(230,81,0,.85)',  borderRadius:4,borderSkipped:false},
      ]},
    options:{responsive:true,maintainAspectRatio:true,
      plugins:{legend:{display:true,labels:{color:'#455A64',font:{size:11},boxWidth:12}}},
      scales:{x:{ticks:{color:'#90A4AE',font:{size:10}},grid:{color:'#EEF3F9'}},
        y:{min:0,ticks:{color:'#90A4AE',font:{size:10},stepSize:1},grid:{color:'#EEF3F9'}}}}});
}
function setModalHand(h){modalHand=h;document.querySelectorAll('.hand-btn').forEach(b=>b.classList.toggle('active',b.dataset.hand===h));updateModalStats();}
function setModalPeriod(n){modalPeriod=n;document.querySelectorAll('.period-btn').forEach(b=>b.classList.toggle('active',String(b.dataset.n)===String(n)));drawHistogram();}

// ── Load ───────────────────────────────────────────────────────────────────
async function loadData() {
  document.getElementById('tableSection').style.display='none';
  document.getElementById('errDiv').style.display='none';
  document.getElementById('loadDiv').style.display='flex';
  try {
    const [dataRes]=await Promise.all([fetch('/api/bets'),loadScores()]);
    const data=await dataRes.json();
    document.getElementById('loadDiv').style.display='none';
    if(data.error){document.getElementById('errMsg').textContent=data.error;document.getElementById('errDiv').style.display='';return;}
    allProj=data.projections||[];
    document.getElementById('tableSection').style.display='';
    applyFilters();
  } catch(e){
    document.getElementById('loadDiv').style.display='none';
    document.getElementById('errMsg').textContent='Failed: '+e.message;
    document.getElementById('errDiv').style.display='';
  }
}
loadData();
setInterval(loadScores, 60000);
</script>
</body>
</html>"""


@app.route("/")
def index(): return HTML


@app.route("/api/bets")
def api_bets():
    try:
        games     = cached(f"games:{today()}", get_todays_games)
        savant_df = cached(f"savant:{today()}", get_savant_xstats, ttl=3600)

        odds_map = {}
        if ODDS_API_KEY:
            events = cached(f"odds_events:{today()}", get_mlb_events, ttl=300)
            for ev in events:
                odds_map.update(aggregate_best_odds(get_event_props(ev["id"])))
                time.sleep(0.1)
            log.info(f"Odds loaded: {len(odds_map)} entries")

        projs, seen = [], set()
        for game in games:
            for side in ("home","away"):
                opp  = "away" if side=="home" else "home"
                opid = game.get(f"{opp}_pitcher_id")
                lu   = game.get(f"{side}_lineup",[])
                if not lu: continue
                _d  = today()
                ph  = cached(f"hand_{opid}:{_d}", lambda i=opid: get_pitcher_hand(i))
                ps  = cached(f"pstat_{opid}:{_d}", lambda i=opid: get_pitcher_stats(i))
                for player in lu:
                    pid = player["id"]
                    if pid in seen: continue
                    seen.add(pid)
                    spl    = cached(f"spl_{pid}:{_d}", lambda i=pid: get_player_splits(i))
                    gl     = cached(f"gl_{pid}:{_d}",  lambda i=pid: get_player_game_log(i))
                    sstats = cached(f"ss_{pid}:{_d}",  lambda i=pid: get_player_season_stats(i))
                    time.sleep(0.03)
                    pid_s = str(pid)
                    xs = {}
                    if pid_s in savant_df.index:
                        row = savant_df.loc[pid_s]
                        xs  = {"xba":str(row.get("xba","")),"xslg":str(row.get("xslg","")),"xwoba":str(row.get("xwoba","")),"barrel":str(row.get("barrel_pct",""))}
                    proj = _build_proj(pid, player["name"], game, side, ph, ps, spl, gl, xs, sstats,
                                       player.get("confirmed",False), odds_map if odds_map else None)
                    projs.append(proj)

        projs.sort(key=lambda x: x["p_hit"], reverse=True)
        return jsonify({"projections": projs, "has_odds": bool(odds_map)})
    except Exception as e:
        log.exception("api_bets failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scores")
def api_scores():
    try:
        r = requests.get(f"{MLB_API}/schedule",
            params={"sportId":1,"date":today(),"hydrate":"linescore,team"}, timeout=15)
        r.raise_for_status()
        scores = []
        for d in r.json().get("dates",[]):
            for g in d.get("games",[]):
                ls=g.get("linescore",{})
                scores.append({
                    "game_id":      g["gamePk"],
                    "status":       g.get("status",{}).get("abstractGameState","Preview"),
                    "inning":       ls.get("currentInning",0),
                    "half":         ls.get("inningHalf",""),
                    "away_team":    g["teams"]["away"]["team"].get("abbreviation",""),
                    "home_team":    g["teams"]["home"]["team"].get("abbreviation",""),
                    "away_score":   g["teams"]["away"].get("score",0),
                    "home_score":   g["teams"]["home"].get("score",0),
                    "away_team_id": g["teams"]["away"]["team"]["id"],
                    "home_team_id": g["teams"]["home"]["team"]["id"],
                    "game_time":    g.get("gameDate",""),
                })
        return jsonify({"scores": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh")
def api_refresh():
    with _cache_lock: _cache.clear()
    return jsonify({"status":"cache cleared"})


if __name__ == "__main__":
    print(f"\n  ⚾  MLB Prop Finder  →  http://localhost:5050")
    print(f"  Date: {today()}  |  Odds key: {'SET ✓' if ODDS_API_KEY else 'NOT SET'}\n")
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)

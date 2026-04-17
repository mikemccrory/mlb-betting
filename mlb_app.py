"""
MLB Prop Finder — Streamlit App
================================
Run with:
    ODDS_API_KEY=your_key streamlit run mlb_app.py
"""

import os, math, time, threading, logging
from io import StringIO
from datetime import date
from typing import Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚾ MLB Prop Finder",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Minimal CSS tweaks ──────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stHeader"] { background: transparent; }
  section[data-testid="stSidebar"] { background: #1e293b; }
  h1,h2,h3,h4 { color: #f1f5f9 !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px; color: #94a3b8;
    padding: 6px 18px; border: 1px solid #334155;
  }
  .stTabs [aria-selected="true"] {
    background: #2563eb !important; color: #fff !important; border-color: #2563eb !important;
  }
  div[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 12px 16px;
  }
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .stButton>button {
    background: #2563eb; color: #fff; border: none;
    border-radius: 8px; padding: 6px 18px; font-weight: 600;
  }
  .stButton>button:hover { background: #1d4ed8; }
  .stSelectbox>div>div, .stTextInput>div>div {
    background: #1e293b !important; color: #e2e8f0 !important;
    border-color: #334155 !important; border-radius: 8px !important;
  }
  label { color: #94a3b8 !important; }
  .prob-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ─── Config ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
MLB_API      = "https://statsapi.mlb.com/api/v1"
ODDS_API     = "https://api.the-odds-api.com/v4"
SAVANT_BASE  = "https://baseballsavant.mlb.com"

AVG_PA_PER_GAME       = 3.8
LEAGUE_AVG_BA         = 0.248
LEAGUE_AVG_HR_PA      = 0.033
LEAGUE_AVG_BARREL_PCT = 8.0
LEAGUE_AVG_XSLG       = 0.411
LEAGUE_AVG_XBA        = 0.248
LEAGUE_AVG_XWOBA      = 0.315

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

PROP_MARKETS = ["batter_hits","batter_home_runs","batter_rbis","batter_total_bases"]
BET_LABELS   = {
    "batter_hits":"Anytime Hit","batter_home_runs":"Anytime HR",
    "batter_rbis":"Anytime RBI","batter_total_bases":"Total Bases 2.5+",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def today() -> str: return date.today().isoformat()

def normalize_name(name: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii","ignore").decode("ascii").lower().replace(".","").replace("-"," ").replace("'","").strip()

def _logit(p):
    p = max(1e-6, min(1-1e-6, p))
    return math.log(p/(1-p))

def _sigmoid(z): return 1/(1+math.exp(-z))
def _log_adj(r):  return max(-1.5, min(1.5, math.log(max(r,1e-4))))

def _poisson_cdf(k, lam):
    r, t = 0.0, math.exp(-lam)
    for i in range(k+1):
        r += t; t *= lam/(i+1)
    return min(r, 1.0)

def american_to_decimal(o): return (o/100+1) if o>=0 else (100/abs(o)+1)
def implied_prob(o):         return (100/(o+100)) if o>=0 else (abs(o)/(abs(o)+100))
def calc_ev(p, o):           return round((p*(american_to_decimal(o)-1)-(1-p))*100, 2)

def _recent_avg(games, n):
    g=games[:n]; ab=sum(x["ab"] for x in g); h=sum(x["h"] for x in g)
    return (h/ab) if ab>=5 else None

def _recent_hr_rate(games, n):
    g=games[:n]; pa=sum(x["ab"]+x.get("bb",0) for x in g); hrs=sum(x["hr"] for x in g)
    return (hrs/pa) if pa>=5 else None

# ─── Probability models ──────────────────────────────────────────────────────
def hit_prob(splits, ph, games, venue, pstats, savant=None):
    split = splits.get("vl" if ph=="L" else "vr", {})
    xba=None
    if savant:
        try: xba=float(savant.get("xba") or "")
        except: pass
    split_avg=None
    try: split_avg=float(split.get("avg") or "")
    except: pass
    base=xba or split_avg or LEAGUE_AVG_XBA
    base=max(0.10,min(0.45,base)); z=_logit(base)
    if split_avg and split.get("pa",0)>=30:
        z+=1.2*_log_adj(split_avg/LEAGUE_AVG_XBA)
    a5=_recent_avg(games,5)
    if a5: z+=1.0*_log_adj(a5/LEAGUE_AVG_BA)
    a10=_recent_avg(games,10)
    if a10: z+=0.6*_log_adj(a10/LEAGUE_AVG_BA)
    z+=0.5*_log_adj(HIT_PARK.get(venue,1.0))
    z+=0.8*_log_adj(pstats.get("avg_against",LEAGUE_AVG_BA)/LEAGUE_AVG_BA)
    r=_sigmoid(z); r=max(0.08,min(0.50,r))
    return round(1-(1-r)**AVG_PA_PER_GAME,4)

def hr_prob(splits, ph, games, venue, pstats, savant=None):
    split=splits.get("vl" if ph=="L" else "vr",{})
    barrel=LEAGUE_AVG_BARREL_PCT; xslg=LEAGUE_AVG_XSLG
    if savant:
        try: barrel=float(savant.get("barrel") or "") or barrel
        except: pass
        try: xslg=float(savant.get("xslg") or "") or xslg
        except: pass
    sp_pa=int(split.get("pa",0) or 0); sp_hr=int(split.get("hr",0) or 0)
    sr=(sp_hr/sp_pa) if sp_pa>=30 else LEAGUE_AVG_HR_PA
    z=_logit(LEAGUE_AVG_HR_PA)
    z+=1.8*_log_adj(barrel/LEAGUE_AVG_BARREL_PCT)
    z+=1.0*_log_adj(xslg/LEAGUE_AVG_XSLG)
    if sp_pa>=30: z+=1.4*_log_adj(sr/LEAGUE_AVG_HR_PA)
    h5=_recent_hr_rate(games,5)
    if h5: z+=1.0*_log_adj((h5+LEAGUE_AVG_HR_PA)/(2*LEAGUE_AVG_HR_PA))
    h10=_recent_hr_rate(games,10)
    if h10: z+=0.7*_log_adj((h10+LEAGUE_AVG_HR_PA)/(2*LEAGUE_AVG_HR_PA))
    z+=0.6*_log_adj(HR_PARK.get(venue,1.0))
    z+=0.8*_log_adj(pstats.get("hr_per_9",1.20)/1.20)
    r=_sigmoid(z); r=max(0.003,min(0.14,r))
    return round(1-(1-r)**AVG_PA_PER_GAME,4)

def total_bases_prob(splits, ph, games, venue, pstats, savant=None, threshold=2.5):
    split=splits.get("vl" if ph=="L" else "vr",{})
    xslg=LEAGUE_AVG_XSLG
    if savant:
        try: xslg=float(savant.get("xslg") or "") or xslg
        except: pass
    slg=split.get("slg",xslg) or xslg
    bl=0.55*xslg+0.45*float(slg)
    exp=bl*AVG_PA_PER_GAME*HR_PARK.get(venue,1.0)*(pstats.get("avg_against",LEAGUE_AVG_BA)/LEAGUE_AVG_BA)
    exp=max(0.1,exp)
    return round(max(0.01,min(0.99,1-_poisson_cdf(int(threshold-0.001),exp))),4)

def rbi_prob(splits, ph, games, venue, pstats, savant=None):
    hp=hit_prob(splits,ph,games,venue,pstats,savant)
    return round(1-(1-(hp/AVG_PA_PER_GAME)*0.28)**AVG_PA_PER_GAME,4)

# ─── MLB API ─────────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=15):
    return requests.get(url, params=params, timeout=timeout)

def get_team_roster(team_id):
    try:
        r=_get(f"{MLB_API}/teams/{team_id}/roster",{"rosterType":"active","season":date.today().year})
        r.raise_for_status()
        return [{"id":p["person"]["id"],"name":p["person"]["fullName"],"confirmed":False}
                for p in r.json().get("roster",[])
                if p.get("position",{}).get("abbreviation","P") not in ("P","SP","RP","CL")]
    except: return []

def get_todays_games():
    r=_get(f"{MLB_API}/schedule",{"sportId":1,"date":today(),"hydrate":"lineups,probablePitcher(note),team,venue"})
    r.raise_for_status()
    games=[]
    for d in r.json().get("dates",[]):
        for g in d.get("games",[]):
            info={
                "game_id":g["gamePk"],"game_time":g.get("gameDate",""),
                "venue":g.get("venue",{}).get("name","Unknown Ballpark"),
                "home_team":g["teams"]["home"]["team"]["name"],
                "away_team":g["teams"]["away"]["team"]["name"],
                "home_team_id":g["teams"]["home"]["team"]["id"],
                "away_team_id":g["teams"]["away"]["team"]["id"],
            }
            for side in ("home","away"):
                pp=g["teams"][side].get("probablePitcher",{})
                info[f"{side}_pitcher_id"]=pp.get("id")
                info[f"{side}_pitcher_name"]=pp.get("fullName","TBD")
            lineups=g.get("lineups",{})
            for side in ("home","away"):
                players=lineups.get(f"{side}Players",[])
                info[f"{side}_lineup"]=[{"id":p["id"],"name":p.get("fullName",""),"confirmed":True} for p in players]
            games.append(info)
    for game in games:
        for side in ("home","away"):
            if not game[f"{side}_lineup"]:
                game[f"{side}_lineup"]=get_team_roster(game[f"{side}_team_id"])
    return games

def get_player_splits(player_id):
    try:
        r=_get(f"{MLB_API}/people/{player_id}/stats",{"stats":"statSplits","season":date.today().year,"group":"hitting","sitCodes":"vl,vr"})
        r.raise_for_status()
        splits={}
        for sp in r.json().get("stats",[{}])[0].get("splits",[]):
            code=sp.get("split",{}).get("code",""); s=sp.get("stat",{})
            pa=int(s.get("plateAppearances",0) or 0); hr=int(s.get("homeRuns",0) or 0)
            splits[code]={"avg":float(s.get("avg",0) or 0),"slg":float(s.get("slg",0) or 0),
                          "ops":float(s.get("ops",0) or 0),"hr":hr,"pa":pa,"hr_rate":hr/max(pa,1)}
        return splits
    except: return {}

def get_player_game_log(player_id, n=10):
    try:
        r=_get(f"{MLB_API}/people/{player_id}/stats",{"stats":"gameLog","season":date.today().year,"group":"hitting"})
        r.raise_for_status()
        raw=sorted(r.json().get("stats",[{}])[0].get("splits",[]),key=lambda x:x.get("date",""),reverse=True)[:n]
        return [{"date":g.get("date",""),"opp":g.get("opponent",{}).get("abbreviation",""),
                 "ab":g["stat"].get("atBats",0),"h":g["stat"].get("hits",0),
                 "hr":g["stat"].get("homeRuns",0),"rbi":g["stat"].get("rbi",0),
                 "bb":g["stat"].get("baseOnBalls",0),"k":g["stat"].get("strikeOuts",0)} for g in raw]
    except: return []

def get_pitcher_hand(pid):
    if not pid: return "R"
    try:
        r=_get(f"{MLB_API}/people/{pid}"); r.raise_for_status()
        return r.json().get("people",[{}])[0].get("pitchHand",{}).get("code","R") or "R"
    except: return "R"

def get_pitcher_stats(pid):
    d={"era":4.50,"whip":1.30,"hr_per_9":1.20,"avg_against":0.250}
    if not pid: return d
    try:
        r=_get(f"{MLB_API}/people/{pid}/stats",{"stats":"season","season":date.today().year,"group":"pitching"})
        r.raise_for_status()
        splits=r.json().get("stats",[{}])[0].get("splits",[])
        if not splits: return d
        s=splits[0].get("stat",{})
        return {"era":float(s.get("era",4.50) or 4.50),"whip":float(s.get("whip",1.30) or 1.30),
                "hr_per_9":float(s.get("homeRunsPer9",1.20) or 1.20),"avg_against":float(s.get("avg",0.250) or 0.250)}
    except: return d

def get_savant_xstats():
    try:
        r=requests.get(f"{SAVANT_BASE}/expected_statistics",
                       params={"type":"batter","year":date.today().year,"position":"","team":"","min":25,"csv":"true"},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=25)
        r.raise_for_status()
        df=pd.read_csv(StringIO(r.text))
        col_map={"player_id":"mlb_id","est_ba":"xba","est_slg":"xslg","est_woba":"xwoba","barrel_batted_rate":"barrel_pct"}
        df=df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        if "mlb_id" not in df.columns: return pd.DataFrame()
        df["mlb_id"]=df["mlb_id"].astype(str)
        return df.set_index("mlb_id")
    except: return pd.DataFrame()

def get_mlb_events():
    if not ODDS_API_KEY: return []
    try:
        r=requests.get(f"{ODDS_API}/sports/baseball_mlb/events",
                       params={"apiKey":ODDS_API_KEY,"dateFormat":"iso"},timeout=12)
        if r.status_code==401: return []
        r.raise_for_status()
        return [e for e in r.json() if e.get("commence_time","")[:10]==today()]
    except: return []

def get_event_props(event_id):
    if not ODDS_API_KEY: return []
    try:
        r=requests.get(f"{ODDS_API}/sports/baseball_mlb/events/{event_id}/odds",
                       params={"apiKey":ODDS_API_KEY,"regions":"us","markets":",".join(PROP_MARKETS),
                               "oddsFormat":"american","bookmakers":"draftkings,fanduel"},timeout=15)
        r.raise_for_status()
        return r.json().get("bookmakers",[])
    except: return []

def aggregate_best_odds(bookmakers):
    best={}
    for book in bookmakers:
        bname=book.get("title","")
        for mkt in book.get("markets",[]):
            mk=mkt.get("key","")
            for oc in mkt.get("outcomes",[]):
                if oc.get("name","").lower() not in ("over","yes"): continue
                player=oc.get("description",oc.get("name",""))
                odds=int(oc.get("price",0) or 0)
                threshold=float(oc.get("point",0.5) or 0.5)
                key=f"{normalize_name(player)}|{mk}"
                if key not in best or odds>best[key]["odds"]:
                    best[key]={"player":player,"market":mk,"odds":odds,"book":bname,"threshold":threshold}
    return best

# ─── Main analysis (Streamlit cached) ────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(_date_key: str):
    """Full pipeline. _date_key busts cache daily."""
    games = get_todays_games()
    if not games:
        return {"bets":[], "projections":[]}

    savant_df = get_savant_xstats()

    events = get_mlb_events()
    odds_map = {}
    for ev in events:
        odds_map.update(aggregate_best_odds(get_event_props(ev["id"])))
        time.sleep(0.05)

    bets, projections, seen = [], [], set()

    for game in games:
        venue = game["venue"]
        for side in ("home","away"):
            opp      = "away" if side=="home" else "home"
            opp_pid  = game.get(f"{opp}_pitcher_id")
            opp_pnm  = game.get(f"{opp}_pitcher_name","TBD")
            lineup   = game.get(f"{side}_lineup",[])
            if not lineup: continue

            phand  = get_pitcher_hand(opp_pid)
            pstats = get_pitcher_stats(opp_pid)

            for player in lineup:
                pid=player["id"]; name=player["name"]
                if pid in seen: continue
                seen.add(pid)
                splits = get_player_splits(pid)
                gl     = get_player_game_log(pid, 10)
                time.sleep(0.03)

                pid_s=str(pid); xstats={}
                if pid_s in savant_df.index:
                    row=savant_df.loc[pid_s]
                    xstats={"xba":str(row.get("xba","")),"xslg":str(row.get("xslg","")),"xwoba":str(row.get("xwoba","")),"barrel":str(row.get("barrel_pct",""))}

                p_hit  = hit_prob(splits,phand,gl,venue,pstats,xstats)
                p_hr   = hr_prob(splits,phand,gl,venue,pstats,xstats)
                p_tb   = total_bases_prob(splits,phand,gl,venue,pstats,xstats)
                p_rbi  = rbi_prob(splits,phand,gl,venue,pstats,xstats)
                split_used = splits.get("vl" if phand=="L" else "vr",{})

                proj = {
                    "player":name,"team":game[f"{side}_team"],"venue":venue,
                    "pitcher":opp_pnm,"pitcher_hand":phand,
                    "p_hit":p_hit,"p_hr":p_hr,"p_tb":p_tb,"p_rbi":p_rbi,
                    "xstats":xstats,"split":split_used,"game_log":gl,
                    "confirmed":player.get("confirmed",False),
                }
                projections.append(proj)

                for mkt,true_p in {"batter_hits":p_hit,"batter_home_runs":p_hr,"batter_total_bases":p_tb,"batter_rbis":p_rbi}.items():
                    key=f"{normalize_name(name)}|{mkt}"
                    if key not in odds_map: continue
                    od=odds_map[key]; amer=od["odds"]; ev=calc_ev(true_p,amer)
                    bets.append({**proj,"bet_type":BET_LABELS[mkt],"market":mkt,
                                 "true_prob":true_p,"implied_prob":round(implied_prob(amer),4),
                                 "odds":amer,"book":od["book"],"ev":ev})

    bets.sort(key=lambda x:x["ev"],reverse=True)
    projections.sort(key=lambda x:x["p_hr"],reverse=True)
    return {"bets":bets,"projections":projections}

# ─── Chart helpers ────────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="#1e293b",plot_bgcolor="#0f172a",font_color="#94a3b8")

def prob_bar_chart(p):
    labels=["P(Hit)","P(HR)","P(TB 2.5+)","P(RBI)"]
    vals=[p["p_hit"]*100, p["p_hr"]*100, p["p_tb"]*100, p["p_rbi"]*100]
    colors=["#34d399","#f97316","#a78bfa","#fbbf24"]
    fig=go.Figure(go.Bar(x=vals,y=labels,orientation="h",marker_color=colors,
                         text=[f"{v:.1f}%" for v in vals],textposition="outside"))
    fig.update_layout(**DARK,height=180,margin=dict(l=10,r=40,t=10,b=10),
                      xaxis=dict(range=[0,110],showgrid=False,showticklabels=False),
                      yaxis=dict(showgrid=False))
    return fig

def form_charts(gl):
    if not gl:
        return None, None, None
    df=pd.DataFrame(gl[::-1])  # chronological
    df["date"]=df["date"].str[5:]  # MM-DD

    fig_h=go.Figure(go.Bar(x=df["date"],y=df["h"],marker_color="#34d399",name="Hits"))
    fig_h.update_layout(**DARK,height=200,margin=dict(l=0,r=0,t=24,b=0),title_text="Hits / game",title_font_color="#64748b",title_font_size=11)

    fig_hr=go.Figure(go.Bar(x=df["date"],y=df["hr"],marker_color="#f97316",name="HR"))
    fig_hr.update_layout(**DARK,height=200,margin=dict(l=0,r=0,t=24,b=0),title_text="HR / game",title_font_color="#64748b",title_font_size=11)

    # Rolling avg
    avgs=[]
    for i in range(len(df)):
        sl=df.iloc[:i+1]; ab=sl["ab"].sum(); h=sl["h"].sum()
        avgs.append(round(h/ab,3) if ab>0 else None)
    fig_avg=go.Figure(go.Scatter(x=df["date"],y=avgs,mode="lines+markers",
                                 line=dict(color="#60a5fa",width=2),marker=dict(size=5,color="#60a5fa"),
                                 fill="tozeroy",fillcolor="rgba(96,165,250,.08)"))
    fig_avg.update_layout(**DARK,height=180,margin=dict(l=0,r=0,t=24,b=0),title_text="Rolling AVG",title_font_color="#64748b",title_font_size=11,
                          yaxis=dict(range=[0,0.5],tickformat=".3f",showgrid=True,gridcolor="#1e293b"))
    return fig_h, fig_hr, fig_avg

def xstats_chart(xs):
    xba=float(xs.get("xba") or 0) or 0
    xslg=float(xs.get("xslg") or 0) or 0
    xwoba=float(xs.get("xwoba") or 0) or 0
    fig=go.Figure()
    cats=["xBA","xSLG","xwOBA"]
    player_vals=[xba,xslg,xwoba]
    lg_vals=[0.248,0.411,0.315]
    fig.add_trace(go.Bar(name="Player",x=cats,y=player_vals,marker_color=["#34d399","#a78bfa","#60a5fa"],text=[f"{v:.3f}" for v in player_vals],textposition="outside"))
    fig.add_trace(go.Bar(name="Lg Avg",x=cats,y=lg_vals,marker_color=["rgba(52,211,153,.2)","rgba(167,139,250,.2)","rgba(96,165,250,.2)"],
                         marker_line_color=["#34d399","#a78bfa","#60a5fa"],marker_line_width=1))
    fig.update_layout(**DARK,height=240,margin=dict(l=0,r=0,t=10,b=0),barmode="group",legend=dict(font_color="#94a3b8"))
    return fig

# ─── Player detail panel ──────────────────────────────────────────────────────
def show_player_detail(p, bets_for_player):
    st.markdown(f"### {p['player']}")
    st.caption(f"{p['team']} · vs **{p['pitcher']}** ({'⬅ LHP' if p['pitcher_hand']=='L' else '➡ RHP'}) · {p['venue']}")

    # Props from DK/FD
    if bets_for_player:
        st.markdown("**Props (DraftKings / FanDuel)**")
        cols = st.columns(len(bets_for_player))
        for col, b in zip(cols, bets_for_player):
            ev_color = "🟢" if b["ev"] >= 0 else "🔴"
            odds_str = f"+{b['odds']}" if b['odds'] >= 0 else str(b['odds'])
            col.metric(
                label=b["bet_type"],
                value=odds_str,
                delta=f"{ev_color} ${b['ev']:+.2f} EV · {b['book']}"
            )
        st.divider()

    tab_ov, tab_form, tab_xstat = st.tabs(["Overview", "Recent Form", "Statcast"])

    with tab_ov:
        # Prob bar chart
        st.plotly_chart(prob_bar_chart(p), use_container_width=True, config={"displayModeBar":False})

        # Splits
        sp = p.get("split", {})
        if sp and sp.get("pa", 0) >= 1:
            st.markdown(f"**vs {'LHP' if p['pitcher_hand']=='L' else 'RHP'} splits**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("AVG", sp.get("avg","—"))
            c2.metric("SLG", sp.get("slg","—"))
            c3.metric("OPS", sp.get("ops","—"))
            c4.metric("HR", f"{sp.get('hr',0)} in {sp.get('pa',0)} PA")
        else:
            st.caption("No split data available")

    with tab_form:
        gl = p.get("game_log", [])
        if gl:
            fh, fhr, favg = form_charts(gl)
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(fh,  use_container_width=True, config={"displayModeBar":False})
            with c2: st.plotly_chart(fhr, use_container_width=True, config={"displayModeBar":False})
            st.plotly_chart(favg, use_container_width=True, config={"displayModeBar":False})
            df_gl = pd.DataFrame(gl)[["date","opp","ab","h","hr","rbi","bb","k"]]
            df_gl.columns = ["Date","Opp","AB","H","HR","RBI","BB","K"]
            st.dataframe(df_gl, use_container_width=True, hide_index=True)
        else:
            st.caption("No game log data")

    with tab_xstat:
        xs = p.get("xstats", {})
        xba  = xs.get("xba","")
        xslg = xs.get("xslg","")
        xwoba= xs.get("xwoba","")
        brl  = xs.get("barrel","")
        has_data = any(v and v not in ("","nan","None") for v in [xba,xslg,xwoba])
        if has_data:
            st.plotly_chart(xstats_chart(xs), use_container_width=True, config={"displayModeBar":False})
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("xBA",   xba   or "—", delta=f"lg {0.248}")
            c2.metric("xSLG",  xslg  or "—", delta=f"lg {0.411}")
            c3.metric("xwOBA", xwoba or "—", delta=f"lg {0.315}")
            if brl and brl not in ("nan","None",""):
                try: c4.metric("Barrel%", f"{float(brl):.1f}%", delta=f"lg 8.0%")
                except: pass
        else:
            st.caption("No Statcast data available")

# ─── Main app ─────────────────────────────────────────────────────────────────
def main():
    # Header
    col_title, col_refresh = st.columns([5,1])
    with col_title:
        st.title("⚾ MLB Prop Finder")
        if not ODDS_API_KEY:
            st.warning("No ODDS_API_KEY set — showing projections only. Run with `ODDS_API_KEY=your_key streamlit run mlb_app.py`")
    with col_refresh:
        st.write("")
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    with st.spinner("Fetching lineups · Baseball Savant · DK/FD odds… (first load takes ~60s)"):
        data = run_analysis(today())

    bets = data["bets"]
    projs = data["projections"]

    if not projs:
        st.error("No games or data found. Check your internet connection.")
        return

    # Summary metrics
    pos_bets = [b for b in bets if b["ev"] > 0]
    games_count = len({p["team"] for p in projs}) // 2 or len({p["team"] for p in projs})
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("+EV Bets", len(pos_bets))
    m2.metric("Best EV / $100", f"${pos_bets[0]['ev']:+.2f}" if pos_bets else "—")
    m3.metric("Games Today", games_count)
    m4.metric("Players Analyzed", len(projs))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_bets, tab_players = st.tabs(["🎯 Bets (DK / FD)", "👥 All Players"])

    # ── BETS TAB ──────────────────────────────────────────────────────────────
    with tab_bets:
        if not bets:
            st.info("No odds available yet — DK/FD post props a few hours before game time.")
        else:
            # Filters
            fc1,fc2,fc3 = st.columns([2,2,2])
            mkt_filter = fc1.selectbox("Market", ["All","Anytime Hit","Anytime HR","Total Bases 2.5+","Anytime RBI"])
            ev_filter  = fc2.selectbox("Min EV", ["0 (+EV only)","Any",">$5",">$10",">$20"])
            sort_opt   = fc3.selectbox("Sort by", ["EV","True Prob","HR Prob"])

            ev_thresholds = {"0 (+EV only)":0,"Any":-999,">$5":5,">$10":10,">$20":20}
            min_ev = ev_thresholds[ev_filter]
            sort_map = {"EV":"ev","True Prob":"true_prob","HR Prob":"p_hr"}
            sort_key = sort_map[sort_opt]

            filtered = [b for b in bets if b["ev"]>=min_ev]
            if mkt_filter!="All": filtered=[b for b in filtered if b["bet_type"]==mkt_filter]
            filtered.sort(key=lambda x:x.get(sort_key,0),reverse=True)

            if not filtered:
                st.info("No bets match your filters.")
            else:
                # Build display dataframe
                rows=[]
                for b in filtered:
                    odds_str=f"+{b['odds']}" if b['odds']>=0 else str(b['odds'])
                    rows.append({
                        "Player":b["player"],"Team":b["team"],"Bet":b["bet_type"],
                        "Pitcher":f"{b['pitcher']} ({'LHP' if b['pitcher_hand']=='L' else 'RHP'})",
                        "Venue":b["venue"][:22]+"…" if len(b["venue"])>22 else b["venue"],
                        "True P":f"{b['true_prob']*100:.1f}%",
                        "Imp P":f"{b['implied_prob']*100:.1f}%",
                        "Odds":odds_str,"Book":b["book"],
                        "EV/$100":f"{'+'if b['ev']>=0 else ''}${b['ev']:.2f}",
                    })
                df_bets=pd.DataFrame(rows)

                # Colour EV column
                def style_ev(val):
                    return "color:#34d399;font-weight:700" if val.startswith("+") else "color:#f87171"

                styled=df_bets.style.applymap(style_ev,subset=["EV/$100"])
                st.dataframe(styled, use_container_width=True, hide_index=True, height=min(600, 40+36*len(rows)))

                # Player selector for detail
                st.divider()
                names=[b["player"] for b in filtered]
                unique_names=list(dict.fromkeys(names))
                sel=st.selectbox("View player detail", ["— select —"]+unique_names, key="bets_sel")
                if sel!="— select —":
                    player_data=next((p for p in projs if p["player"]==sel),None) or next((b for b in bets if b["player"]==sel),None)
                    bets_for = [b for b in bets if b["player"]==sel]
                    if player_data:
                        with st.container():
                            show_player_detail(player_data, bets_for)

    # ── PLAYERS TAB ───────────────────────────────────────────────────────────
    with tab_players:
        pc1,pc2 = st.columns([3,2])
        search = pc1.text_input("Search player or team", placeholder="e.g. Judge, Yankees…")
        psort  = pc2.selectbox("Sort by", ["HR Prob","Hit Prob","TB Prob","RBI Prob"], key="psort")

        psort_map={"HR Prob":"p_hr","Hit Prob":"p_hit","TB Prob":"p_tb","RBI Prob":"p_rbi"}
        psk=psort_map[psort]
        filtered_p=projs
        if search:
            q=search.lower()
            filtered_p=[p for p in projs if q in p["player"].lower() or q in p["team"].lower()]
        filtered_p=sorted(filtered_p,key=lambda x:x.get(psk,0),reverse=True)

        rows_p=[]
        for p in filtered_p:
            xs=p.get("xstats",{})
            rows_p.append({
                "Player":p["player"]+("" if p.get("confirmed") else " ◦"),
                "Team":p["team"],
                "vs Pitcher":f"{p['pitcher']} ({'LHP' if p['pitcher_hand']=='L' else 'RHP'})",
                "P(Hit)":f"{p['p_hit']*100:.1f}%",
                "P(HR)":f"{p['p_hr']*100:.1f}%",
                "P(TB)":f"{p['p_tb']*100:.1f}%",
                "P(RBI)":f"{p['p_rbi']*100:.1f}%",
                "xBA":xs.get("xba",""),"xSLG":xs.get("xslg",""),
            })

        if rows_p:
            st.caption("◦ = roster fallback (lineup not yet official)")
            df_p=pd.DataFrame(rows_p)
            st.dataframe(df_p, use_container_width=True, hide_index=True, height=min(600,40+36*len(rows_p)))

        st.divider()
        pnames=[p["player"] for p in filtered_p]
        psel=st.selectbox("View player detail", ["— select —"]+pnames, key="proj_sel")
        if psel!="— select —":
            pdata=next((p for p in projs if p["player"]==psel),None)
            pbets=[b for b in bets if b["player"]==psel]
            if pdata:
                show_player_detail(pdata, pbets)

if __name__ == "__main__":
    main()

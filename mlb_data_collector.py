#!/usr/bin/env python3
"""
mlb_data_collector.py
=====================
Collects historical batter-game rows for training the hit/HR probability model.

Each row = one batter in one game, with:
  - Pre-game features: season stats, Savant xBA/barrel%, pitcher stats, park, lineup spot
  - Labels: got_hit (1/0), hit_hr (1/0)

Usage:
    python mlb_data_collector.py --start 2024-04-01 --end 2024-09-30
    python mlb_data_collector.py --start 2023-04-01 --end 2023-10-01 --out data/train_2023.csv
    python mlb_data_collector.py --start 2024-04-01 --end 2024-09-30 --append --out data/train.csv

Notes:
  - Uses season-level stats as features (not game-date rolling stats). This is an
    approximation; early-season rows will have small-sample noise.
  - Savant data (xBA, barrel%) is fetched once per season year and cached locally.
  - Rate-limited to ~40 req/min to avoid hitting MLB Stats API limits.
  - Rows are written incrementally so you can Ctrl-C and resume with --append.
"""

import argparse
import csv
import os
import time
import logging
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MLB_API     = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# Park factors (same as app)
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

OUTPUT_COLUMNS = [
    # Identifiers
    "date","game_pk","batter_id","batter_name","team",
    "pitcher_id","pitcher_name","pitcher_hand","lineup_spot","venue",
    # Batter Savant features
    "xba","xslg","barrel_pct","exit_velo",
    # Batter season split vs pitcher hand
    "split_ba","split_slg","split_hr_rate","split_pa",
    # Batter overall season
    "season_ba",
    # Pitcher season features
    "pitcher_avg_against","pitcher_k_per_9","pitcher_hr_per_9","pitcher_era","pitcher_whip",
    # Park
    "park_hit_factor","park_hr_factor",
    # Labels
    "actual_hits","actual_hr",
    "got_hit","hit_hr",
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({"User-Agent": "mlb-model-trainer/1.0"})

def _get(url: str, params=None, retries: int = 3, delay: float = 1.5):
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise e

def _safe(d, key, fallback=None):
    try:
        v = d.get(key) if isinstance(d, dict) else getattr(d, key, None)
        if v in (None, "", "nan", "None", float("nan")):
            return fallback
        return float(v)
    except:
        return fallback

# ── Savant cache (one fetch per season) ──────────────────────────────────────
_savant_cache: Dict[int, pd.DataFrame] = {}

def get_savant(year: int) -> pd.DataFrame:
    if year in _savant_cache:
        return _savant_cache[year]
    log.info(f"Fetching Baseball Savant data for {year}…")
    try:
        r = requests.get(f"{SAVANT_BASE}/expected_statistics", params={
            "type":"batter","year":year,"position":"","team":"","min":25,"csv":"true"
        }, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        col_map = {
            "player_id":"mlb_id","est_ba":"xba","est_slg":"xslg",
            "barrel_batted_rate":"barrel_pct","exit_velocity_avg":"exit_velo",
            "avg_hit_speed":"exit_velo","ba":"season_ba",
        }
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        if "mlb_id" not in df.columns:
            _savant_cache[year] = pd.DataFrame()
            return pd.DataFrame()
        df["mlb_id"] = df["mlb_id"].astype(str)
        df = df.set_index("mlb_id")
        _savant_cache[year] = df
        return df
    except Exception as e:
        log.warning(f"Savant {year} failed: {e}")
        _savant_cache[year] = pd.DataFrame()
        return pd.DataFrame()

# ── Per-player caches (reset each day to save memory) ────────────────────────
_splits_cache:  Dict[str, Dict] = {}
_pitcher_cache: Dict[str, Dict] = {}
_hand_cache:    Dict[str, str]  = {}

def get_splits(player_id: int, year: int) -> Dict:
    key = f"{player_id}_{year}"
    if key in _splits_cache:
        return _splits_cache[key]
    try:
        r = _get(f"{MLB_API}/people/{player_id}/stats",
                 {"stats":"statSplits","season":year,"group":"hitting","sitCodes":"vl,vr"})
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
                "pa": pa, "ab": ab, "hr": hr,
                "hr_rate": hr / max(ab, 1),
            }
        _splits_cache[key] = splits
        return splits
    except:
        return {}

def get_pitcher_hand(pitcher_id: int) -> str:
    key = str(pitcher_id)
    if key in _hand_cache:
        return _hand_cache[key]
    try:
        r = _get(f"{MLB_API}/people/{pitcher_id}")
        hand = r.json().get("people",[{}])[0].get("pitchHand",{}).get("code","R") or "R"
        _hand_cache[key] = hand
        return hand
    except:
        return "R"

def get_pitcher_stats(pitcher_id: int, year: int) -> Dict:
    key = f"{pitcher_id}_{year}"
    if key in _pitcher_cache:
        return _pitcher_cache[key]
    base = {"avg_against":0.250,"k_per_9":8.0,"hr_per_9":1.20,"era":4.50,"whip":1.30}
    try:
        r = _get(f"{MLB_API}/people/{pitcher_id}/stats",
                 {"stats":"season","season":year,"group":"pitching"})
        sp = r.json().get("stats",[{}])[0].get("splits",[])
        if sp:
            s = sp[0].get("stat",{})
            base.update({
                "avg_against": float(s.get("avg",0.250) or 0.250),
                "k_per_9":     float(s.get("strikeoutsPer9Inn",8.0) or 8.0),
                "hr_per_9":    float(s.get("homeRunsPer9",1.20) or 1.20),
                "era":         float(s.get("era",4.50) or 4.50),
                "whip":        float(s.get("whip",1.30) or 1.30),
            })
    except:
        pass
    _pitcher_cache[key] = base
    return base

# ── Per-date data ──────────────────────────────────────────────────────────────
def get_games_for_date(date_str: str) -> List[Dict]:
    """Return list of {game_pk, venue, home_team_id, away_team_id, home_pitcher_id,
    away_pitcher_id, home_pitcher_name, away_pitcher_name} for every game on date_str."""
    try:
        r = _get(f"{MLB_API}/schedule", {
            "sportId": 1, "date": date_str,
            "hydrate": "probablePitcher,team,venue,lineups",
        })
        games = []
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("abstractGameState", "")
                if status not in ("Final", "Live"):
                    continue  # skip not-yet-played games
                info = {
                    "game_pk": g["gamePk"],
                    "venue":   g.get("venue", {}).get("name", "Unknown"),
                }
                for side in ("home", "away"):
                    pp = g["teams"][side].get("probablePitcher", {})
                    info[f"{side}_pitcher_id"]   = pp.get("id")
                    info[f"{side}_pitcher_name"] = pp.get("fullName", "Unknown")
                games.append(info)
        return games
    except Exception as e:
        log.warning(f"Schedule {date_str}: {e}")
        return []

def get_boxscore(game_pk: int) -> Optional[Dict]:
    """Return raw boxscore JSON."""
    try:
        r = _get(f"{MLB_API}/game/{game_pk}/boxscore")
        return r.json()
    except Exception as e:
        log.warning(f"Boxscore {game_pk}: {e}")
        return None

def parse_boxscore(box: Dict, game_pk: int, game_date: str, venue: str,
                   home_pitcher_id, away_pitcher_id,
                   home_pitcher_name: str, away_pitcher_name: str,
                   savant_df: pd.DataFrame) -> List[Dict]:
    """
    Extract one row per batter from a boxscore.
    Returns list of dicts matching OUTPUT_COLUMNS.
    """
    rows = []
    year = int(game_date[:4])

    teams = box.get("teams", {})
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team_name = team_data.get("team", {}).get("name", "")

        # Opposing pitcher (who this batter faced)
        opp_pid  = away_pitcher_id if side == "home" else home_pitcher_id
        opp_pnm  = away_pitcher_name if side == "home" else home_pitcher_name

        if not opp_pid:
            continue

        p_hand   = get_pitcher_hand(opp_pid)
        p_stats  = get_pitcher_stats(opp_pid, year)

        # Batting order: batters listed in battingOrder
        batting_order = team_data.get("battingOrder", [])
        # battingOrder is a list of player IDs in order
        spot_map = {pid: idx + 1 for idx, pid in enumerate(batting_order)}

        players = team_data.get("players", {})
        for pkey, pdata in players.items():
            pos = pdata.get("position", {}).get("abbreviation", "P")
            if pos in ("P", "SP", "RP", "CL"):
                continue

            person  = pdata.get("person", {})
            pid     = person.get("id")
            pname   = person.get("fullName", "")
            stats_s = pdata.get("stats", {}).get("batting", {}).get("summary","")
            stat    = pdata.get("stats", {}).get("batting", {})

            if not pid:
                continue

            actual_hits = int(stat.get("hits", 0) or 0)
            actual_ab   = int(stat.get("atBats", 0) or 0)
            actual_hr   = int(stat.get("homeRuns", 0) or 0)

            # Skip players with no at-bats (pinch runner only, etc.)
            if actual_ab == 0:
                continue

            lineup_spot = spot_map.get(pid, 0)

            # Savant features
            pid_s   = str(pid)
            xba     = None; xslg = None; barrel_pct = None; exit_velo = None; season_ba = None
            if not savant_df.empty and pid_s in savant_df.index:
                row = savant_df.loc[pid_s]
                xba        = _safe(row, "xba")
                xslg       = _safe(row, "xslg")
                barrel_pct = _safe(row, "barrel_pct")
                exit_velo  = _safe(row, "exit_velo")
                season_ba  = _safe(row, "season_ba")

            # Split stats vs this pitcher hand
            splits    = get_splits(pid, year)
            split_key = "vl" if p_hand == "L" else "vr"
            split     = splits.get(split_key, {})
            split_ba       = split.get("avg") or None
            split_slg      = split.get("slg") or None
            split_hr_rate  = split.get("hr_rate") or None
            split_pa       = split.get("pa") or 0

            rows.append({
                "date":         game_date,
                "game_pk":      game_pk,
                "batter_id":    pid,
                "batter_name":  pname,
                "team":         team_name,
                "pitcher_id":   opp_pid,
                "pitcher_name": opp_pnm,
                "pitcher_hand": p_hand,
                "lineup_spot":  lineup_spot,
                "venue":        venue,
                # Savant
                "xba":          xba,
                "xslg":         xslg,
                "barrel_pct":   barrel_pct,
                "exit_velo":    exit_velo,
                # Splits
                "split_ba":     split_ba,
                "split_slg":    split_slg,
                "split_hr_rate":split_hr_rate,
                "split_pa":     split_pa,
                # Season
                "season_ba":    season_ba,
                # Pitcher
                "pitcher_avg_against": p_stats["avg_against"],
                "pitcher_k_per_9":     p_stats["k_per_9"],
                "pitcher_hr_per_9":    p_stats["hr_per_9"],
                "pitcher_era":         p_stats["era"],
                "pitcher_whip":        p_stats["whip"],
                # Park
                "park_hit_factor": HIT_PARK.get(venue, 1.0),
                "park_hr_factor":  HR_PARK.get(venue, 1.0),
                # Labels
                "actual_hits": actual_hits,
                "actual_hr":   actual_hr,
                "got_hit":     1 if actual_hits >= 1 else 0,
                "hit_hr":      1 if actual_hr   >= 1 else 0,
            })

        time.sleep(0.05)  # small pause between player stat calls

    return rows

# ── Date range helpers ────────────────────────────────────────────────────────
def date_range(start: str, end: str):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)

def load_existing_keys(path: str) -> set:
    """Return set of (date, game_pk, batter_id) already in the output file."""
    keys = set()
    if not os.path.exists(path):
        return keys
    try:
        df = pd.read_csv(path, usecols=["date","game_pk","batter_id"])
        for _, row in df.iterrows():
            keys.add((str(row["date"]), int(row["game_pk"]), int(row["batter_id"])))
    except Exception as e:
        log.warning(f"Could not read existing file for dedup: {e}")
    return keys

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect historical MLB batter-game training rows.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out",   default="mlb_training_data.csv", help="Output CSV path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file (skips already-collected game/batter rows)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds to wait between game requests (default 0.3)")
    args = parser.parse_args()

    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Dedup keys if appending
    existing_keys = load_existing_keys(out_path) if args.append else set()

    write_mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(out_path))

    total_rows = 0

    with open(out_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for day in date_range(args.start, args.end):
            year = int(day[:4])
            savant_df = get_savant(year)

            games = get_games_for_date(day)
            if not games:
                log.info(f"{day}: no completed games")
                time.sleep(args.delay)
                continue

            log.info(f"{day}: {len(games)} game(s)")

            for game in games:
                gid = game["game_pk"]
                try:
                    box = get_boxscore(gid)
                    if not box:
                        continue

                    new_rows = parse_boxscore(
                        box, gid, day, game["venue"],
                        game.get("home_pitcher_id"), game.get("away_pitcher_id"),
                        game.get("home_pitcher_name",""), game.get("away_pitcher_name",""),
                        savant_df,
                    )

                    written = 0
                    for row in new_rows:
                        key = (row["date"], row["game_pk"], row["batter_id"])
                        if key in existing_keys:
                            continue
                        writer.writerow(row)
                        existing_keys.add(key)
                        written += 1

                    total_rows += written
                    log.info(f"  game {gid}: {written} batter rows  (total so far: {total_rows})")

                except Exception as e:
                    log.warning(f"  game {gid} failed: {e}")

                time.sleep(args.delay)

    log.info(f"\nDone. {total_rows} rows written to {out_path}")
    log.info("Next: train model with  python mlb_train_model.py --data " + out_path)

if __name__ == "__main__":
    main()

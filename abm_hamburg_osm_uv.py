#!/usr/bin/env -S uv run --script --python 3.11
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "osmnx>=2.0",
#   "networkx>=3.2",
#   "numpy>=1.26",
#   "scikit-learn>=1.3",
# ]
# ///
"""
EMS-ABM Hamburg mit echtem OSM-Routing (uv-ready)

- Lädt das 'drive'-Straßennetz über OSMnx (mit Caching auf GraphML).
- Berechnet Fahrzeiten über Kantenlaengen & Geschwindigkeiten (travel_time).
- Dispatch: wählt je Einsatz die Einheit mit minimaler ETA (Verfügbarkeit + Fahrzeit).
- Performance: k-Kandidatenfilter (Luftlinie) + (u,v)->time Cache.
- Output: Kennzahlen der Fahrtdauer (Median/P90/P95/P99).

Nutzung:
    ./abm_hamburg_osm_uv.py \
        --n-incidents 100000 \
        --n-units 30 \
        --service-time-min 20 \
        --k-candidates 5 \
        --place "Hamburg, Germany" \
        --graphml hamburg_drive.graphml

Hinweise:
- Erstlauf lädt das Netz (Dauer: Minuten) und speichert es als GraphML (Cache).
- Ersetze optional Basen durch reale Wachen via --bases-csv (lat,lon je Zeile).
"""

from __future__ import annotations
from dataclasses import dataclass
import argparse
import math
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import networkx as nx
import osmnx as ox


# -----------------------
# Utility
# -----------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# -----------------------
# Params
# -----------------------
@dataclass
class Hotspot:
    lat: float
    lon: float
    sd_km: float
    weight: float

@dataclass
class SimParams:
    # Raum
    bbox: Tuple[float, float, float, float] = (53.35, 9.70, 53.75, 10.30)
    place_name: str = "Hamburg, Germany"
    graphml_path: Optional[str] = "hamburg_drive.graphml"
    # Nachfrage
    n_incidents: int = 100_000
    seed: int = 42
    # Flotte
    n_units: int = 30
    bases: Optional[List[Tuple[float, float]]] = None
    # Service
    service_time_min: float = 20.0
    # Routing
    hwy_speeds: Optional[dict] = None      # z.B. {"residential": 30, "primary": 50, ...}
    speed_fallback_kph: Optional[float] = 30.0
    k_candidates: int = 5                   # exakte Routen nur für k nächste RTW
    # Hotspots (synthetisch; ersetzbar)
    hotspots: Optional[List[Hotspot]] = None


# -----------------------
# OSM Routing Backend
# -----------------------
class OSMRouter:
    """
    Kapselt OSMnx/NetworkX Graph, bietet:
    - nearest_node(lat, lon) -> node id
    - tt_seconds(u, v) -> Fahrzeit (Sekunden) via Dijkstra auf 'travel_time'
    - einfacher Cache für (u,v)
    """
    def __init__(self, place_name: str, graphml_path: Optional[str],
                 hwy_speeds: Optional[dict], speed_fallback_kph: Optional[float]):
        ox.settings.use_cache = True
        ox.settings.log_console = False

        if graphml_path and os.path.exists(graphml_path):
            G = ox.load_graphml(graphml_path)
        else:
            # 'drive'-Netz laden und vorbereiten
            G = ox.graph_from_place(place_name, network_type="drive", simplify=True, retain_all=False)
            G = ox.routing.add_edge_speeds(G, hwy_speeds=hwy_speeds, fallback=speed_fallback_kph)
            G = ox.routing.add_edge_travel_times(G)
            if graphml_path:
                ox.save_graphml(G, graphml_path)

        self.G: nx.MultiDiGraph = G
        self._tt_cache: Dict[Tuple[int, int], float] = {}

    def nearest_node(self, lat: float, lon: float) -> int:
        # OSMnx erwartet X=lon, Y=lat
        return ox.distance.nearest_nodes(self.G, X=lon, Y=lat)

    def tt_seconds(self, u: int, v: int) -> float:
        if u == v:
            return 0.0
        key = (u, v)
        if key in self._tt_cache:
            return self._tt_cache[key]
        try:
            t = nx.shortest_path_length(self.G, u, v, weight="travel_time")
        except nx.NetworkXNoPath:
            t = float("inf")
        self._tt_cache[key] = t
        return t


# -----------------------
# Agents
# -----------------------
@dataclass
class Ambulance:
    id: int
    base_latlon: Tuple[float, float]
    node: int
    latlon: Tuple[float, float]
    free_at: float  # Minuten


# -----------------------
# Incident generator
# -----------------------
class IncidentStream:
    def __init__(self, params: SimParams):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.year_minutes = 365 * 24 * 60
        if self.p.hotspots is None:
            self.p.hotspots = [
                Hotspot(53.5511, 9.9937, 2.0, 0.35),  # Innenstadt
                Hotspot(53.5550, 9.9350, 1.8, 0.20),  # Altona
                Hotspot(53.5700, 10.1000, 2.2, 0.30), # Wandsbek
                Hotspot(53.4600, 9.9800, 2.5, 0.15),  # Harburg
            ]
        total_w = sum(h.weight for h in self.p.hotspots)
        for h in self.p.hotspots:
            h.weight /= total_w
        self.lambda_per_min = self.p.n_incidents / self.year_minutes

    def _bounded(self, lat: float, lon: float) -> Tuple[float, float]:
        min_lat, min_lon, max_lat, max_lon = self.p.bbox
        return (min(max(lat, min_lat), max_lat),
                min(max(lon, min_lon), max_lon))

    def sample_incident_times(self) -> np.ndarray:
        times = []
        t = 0.0
        # Homogener Poisson-Prozess
        while t < self.year_minutes and len(times) < self.p.n_incidents:
            dt = self.rng.exponential(1.0 / self.lambda_per_min)
            t += dt
            if t <= self.year_minutes:
                times.append(t)
        if len(times) < self.p.n_incidents:
            extra = self.p.n_incidents - len(times)
            times += list(self.rng.uniform(0, self.year_minutes, size=extra))
        return np.sort(np.array(times)[:self.p.n_incidents])

    def sample_incident_locations(self, n: int) -> np.ndarray:
        hs = self.p.hotspots
        weights = np.array([h.weight for h in hs], dtype=float)
        comp = self.rng.choice(len(hs), size=n, p=weights)
        lats = np.empty(n); lons = np.empty(n)
        for i, k in enumerate(comp):
            h = hs[k]
            dx_km = self.rng.normal(0.0, h.sd_km)
            dy_km = self.rng.normal(0.0, h.sd_km)
            deg_lat = dy_km / 111.0
            deg_lon = dx_km / (111.0 * math.cos(math.radians(h.lat)))
            lat, lon = self._bounded(h.lat + deg_lat, h.lon + deg_lon)
            lats[i], lons[i] = lat, lon
        return np.column_stack([lats, lons])


# -----------------------
# Model core
# -----------------------
class EMSModel:
    def __init__(self, params: SimParams):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.inc_gen = IncidentStream(self.p)
        self.router = OSMRouter(
            place_name=self.p.place_name,
            graphml_path=self.p.graphml_path,
            hwy_speeds=self.p.hwy_speeds,
            speed_fallback_kph=self.p.speed_fallback_kph,
        )
        self.ambulances: List[Ambulance] = []

    def _init_bases(self) -> List[Tuple[float, float]]:
        if self.p.bases is not None:
            assert len(self.p.bases) >= self.p.n_units, "Not enough bases for n_units"
            return self.p.bases[:self.p.n_units]
        # Gleichmäßig verteilte synthetische Basen in der BBox
        min_lat, min_lon, max_lat, max_lon = self.p.bbox
        n = self.p.n_units
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        lats = np.linspace(min_lat + 0.1*(max_lat-min_lat),
                           max_lat - 0.1*(max_lat-min_lat), rows)
        lons = np.linspace(min_lon + 0.1*(max_lon-min_lon),
                           max_lon - 0.1*(max_lon-min_lon), cols)
        return [(float(lat), float(lon)) for lat in lats for lon in lons][:n]

    def reset(self):
        bases = self._init_bases()
        self.ambulances = []
        for i in range(self.p.n_units):
            lat, lon = bases[i]
            node = self.router.nearest_node(lat, lon)
            self.ambulances.append(
                Ambulance(
                    id=i,
                    base_latlon=(lat, lon),
                    node=node,
                    latlon=(lat, lon),
                    free_at=0.0
                )
            )

    def _candidate_units(self, lat: float, lon: float, k: int) -> List[Ambulance]:
        # k nächstliegende RTW nach Luftlinie (schnell)
        dists = []
        for u in self.ambulances:
            d = haversine_km(lat, lon, u.latlon[0], u.latlon[1])
            dists.append((d, u))
        dists.sort(key=lambda x: x[0])
        return [u for _, u in dists[:min(k, len(dists))]]

    def run(self) -> dict:
        self.reset()
        times = self.inc_gen.sample_incident_times()
        locs = self.inc_gen.sample_incident_locations(len(times))

        travel_times_min = np.empty(len(times), dtype=float)

        for idx, (t_inc, (lat, lon)) in enumerate(zip(times, locs)):
            target_node = self.router.nearest_node(lat, lon)
            candidates = self._candidate_units(lat, lon, self.p.k_candidates)

            best_eta = float('inf'); best_unit = None
            best_t_travel_min = None

            # Prüfe Kandidaten: ETA = depart + travel_time
            for u in candidates:
                depart = max(t_inc, u.free_at)
                tt_sec = self.router.tt_seconds(u.node, target_node)
                if not math.isfinite(tt_sec):
                    continue
                eta = depart + tt_sec/60.0
                if eta < best_eta:
                    best_eta = eta
                    best_unit = u
                    best_t_travel_min = tt_sec/60.0

            # Fallback: falls keiner Route fand, prüfe alle
            if best_unit is None:
                for u in self.ambulances:
                    depart = max(t_inc, u.free_at)
                    tt_sec = self.router.tt_seconds(u.node, target_node)
                    if not math.isfinite(tt_sec):
                        continue
                    eta = depart + tt_sec/60.0
                    if eta < best_eta:
                        best_eta = eta; best_unit = u; best_t_travel_min = tt_sec/60.0

            if best_unit is None:
                travel_times_min[idx] = float('nan')
                continue

            # Metrik
            travel_times_min[idx] = best_t_travel_min

            # Status-Update der gewählten Einheit
            arrival = max(t_inc, best_unit.free_at) + best_t_travel_min
            best_unit.node = target_node
            best_unit.latlon = (lat, lon)
            best_unit.free_at = arrival + self.p.service_time_min

        # Summary
        valid = np.isfinite(travel_times_min)
        tt = travel_times_min[valid]
        summary = {
            "n_incidents": int(valid.sum()),
            "n_units": self.p.n_units,
            "service_time_min": self.p.service_time_min,
            "k_candidates": self.p.k_candidates,
            "travel_time_mean_min": float(np.mean(tt)) if len(tt) else float("nan"),
            "travel_time_std_min": float(np.std(tt)) if len(tt) else float("nan"),
            "travel_time_p50_min": float(np.percentile(tt, 50)) if len(tt) else float("nan"),
            "travel_time_p90_min": float(np.percentile(tt, 90)) if len(tt) else float("nan"),
            "travel_time_p95_min": float(np.percentile(tt, 95)) if len(tt) else float("nan"),
            "travel_time_p99_min": float(np.percentile(tt, 99)) if len(tt) else float("nan"),
        }
        return summary


# -----------------------
# CLI + main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMS-ABM Hamburg mit OSM-Routing (uv-ready)")
    p.add_argument("--n-incidents", type=int, default=100_000)
    p.add_argument("--n-units", type=int, default=30)
    p.add_argument("--service-time-min", type=float, default=20.0)
    p.add_argument("--k-candidates", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--place", type=str, default="Hamburg, Germany")
    p.add_argument("--graphml", type=str, default="hamburg_drive.graphml",
                   help="Pfad zur GraphML-Cachedatei (wird erstellt, falls nicht vorhanden).")
    p.add_argument("--bases-csv", type=str, default=None,
                   help="CSV mit Spalten lat,lon (ohne Header), eine Wache pro Zeile.")
    return p.parse_args()

def load_bases_csv(path: str) -> List[Tuple[float, float]]:
    bases: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lat_str, lon_str = line.split(",")
            bases.append((float(lat_str), float(lon_str)))
    return bases

def main():
    args = parse_args()
    bases = load_bases_csv(args.bases_csv) if args.bases_csv else None

    params = SimParams(
        n_incidents=args.n_incidents,
        n_units=args.n_units,
        service_time_min=args.service_time_min,
        k_candidates=args.k_candidates,
        seed=args.seed,
        place_name=args.place,
        graphml_path=args.graphml,
        bases=bases,
        # Optional: überschreibe innerorts-Regeln selbst:
        # hwy_speeds={"motorway": 90, "trunk": 70, "primary": 50, "secondary": 45,
        #             "tertiary": 40, "residential": 30, "living_street": 10},
        speed_fallback_kph=30.0,
    )

    model = EMSModel(params)
    out = model.run()

    print("=== EMS ABM mit OSM-Routing (Hamburg) — uv ===")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
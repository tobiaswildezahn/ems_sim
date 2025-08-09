# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "streamlit>=1.32",
#   "osmnx>=2.0",
#   "networkx>=3.2",
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "scikit-learn>=1.3"
# ]
# ///
"""
Streamlit-Dashboard für EMS-ABM Hamburg mit OSM-Routing (Echtzeit-Updates)

- Lädt 'drive'-Netz über OSMnx und cached als GraphML.
- Berechnet reale Fahrzeiten (edge length + speed_kph -> travel_time).
- Simuliert Einsätze in Batches und streamt Kennzahlen live ins UI.
- Start lokal mit `uv run streamlit run abm_dashboard.py`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math, os, time

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import streamlit as st

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

def haversine_km_vec(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vektorisierte Haversine-Distanzberechnung."""
    R = 6371.0088
    phi1 = np.radians(lat)
    phi2 = np.radians(lats)
    dphi = phi2 - phi1
    dlmb = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

@dataclass
class Hotspot:
    lat: float
    lon: float
    sd_km: float
    weight: float

@dataclass
class SimParams:
    bbox: Tuple[float, float, float, float] = (53.35, 9.70, 53.75, 10.30)
    place_name: str = "Hamburg, Germany"
    graphml_path: Optional[str] = "hamburg_drive.graphml"
    n_incidents: int = 50_000
    seed: int = 42
    n_units: int = 30
    bases: Optional[List[Tuple[float, float]]] = None
    service_time_min: float = 20.0
    hwy_speeds: Optional[dict] = None
    speed_fallback_kph: Optional[float] = 30.0
    k_candidates: int = 5
    hotspots: Optional[List[Hotspot]] = None

# -----------------------
# OSM Routing Backend (OSMnx v2)
# -----------------------
class OSMRouter:
    def __init__(self, place_name: str, graphml_path: Optional[str],
                 hwy_speeds: Optional[dict], speed_fallback_kph: Optional[float]):
        ox.settings.use_cache = True
        ox.settings.log_console = False

        io = ox.io if hasattr(ox, "io") else ox
        graph = ox.graph if hasattr(ox, "graph") else ox
        routing = ox.routing if hasattr(ox, "routing") else None

        if graphml_path and os.path.exists(graphml_path):
            G = io.load_graphml(graphml_path)
        else:
            with st.spinner("Lade Straßennetz von OSM (einmalig, kann Minuten dauern)…"):
                G = graph.graph_from_place(place_name, network_type="drive", simplify=True, retain_all=False)
                if routing is not None and hasattr(routing, "add_edge_speeds"):
                    G = routing.add_edge_speeds(G, hwy_speeds=hwy_speeds, fallback=speed_fallback_kph)
                    G = routing.add_edge_travel_times(G)
                else:
                    # Fallback für ältere OSMnx-Versionen (1.x)
                    G = ox.speed.add_edge_speeds(G, hwy_speeds=hwy_speeds, fallback=speed_fallback_kph)
                    G = ox.speed.add_edge_travel_times(G)
                if graphml_path:
                    io.save_graphml(G, graphml_path)

        self.G: nx.MultiDiGraph = G
        self._tt_cache: Dict[Tuple[int, int], float] = {}

    def nearest_node(self, lat: float, lon: float) -> int:
        # Achtung: X=lon, Y=lat
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
# Agents & Incident Stream
# -----------------------
@dataclass
class Ambulance:
    id: int
    base_latlon: Tuple[float, float]
    node: int
    latlon: Tuple[float, float]
    free_at: float  # Minuten

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
        n = self.p.n_incidents
        inter = self.rng.exponential(1.0 / self.lambda_per_min, size=int(n * 1.2))
        times = np.cumsum(inter)
        times = times[times <= self.year_minutes]
        if len(times) < n:
            extra = self.rng.uniform(0, self.year_minutes, size=n - len(times))
            times = np.concatenate([times, extra])
        return np.sort(times)[:n]

    def sample_incident_locations(self, n: int) -> np.ndarray:
        hs = self.p.hotspots
        weights = np.array([h.weight for h in hs], dtype=float)
        comp = self.rng.choice(len(hs), size=n, p=weights)
        lats = np.empty(n); lons = np.empty(n)
        for idx, h in enumerate(hs):
            mask = comp == idx
            cnt = mask.sum()
            if cnt == 0:
                continue
            dx = self.rng.normal(0.0, h.sd_km, size=cnt)
            dy = self.rng.normal(0.0, h.sd_km, size=cnt)
            deg_lat = dy / 111.0
            deg_lon = dx / (111.0 * math.cos(math.radians(h.lat)))
            lat = h.lat + deg_lat
            lon = h.lon + deg_lon
            min_lat, min_lon, max_lat, max_lon = self.p.bbox
            lats[mask] = np.clip(lat, min_lat, max_lat)
            lons[mask] = np.clip(lon, min_lon, max_lon)
        return np.column_stack([lats, lons])

# -----------------------
# Model core (mit Streaming)
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
            self.ambulances.append(Ambulance(
                id=i, base_latlon=(lat, lon), node=node, latlon=(lat, lon), free_at=0.0
            ))

    def _candidate_units(self, lat: float, lon: float, k: int) -> List[Ambulance]:
        coords = np.array([u.latlon for u in self.ambulances])
        dists = haversine_km_vec(lat, lon, coords[:,0], coords[:,1])
        if len(dists) <= k:
            idx = np.argsort(dists)
        else:
            idx = np.argpartition(dists, k)[:k]
            idx = idx[np.argsort(dists[idx])]
        return [self.ambulances[i] for i in idx]

    def _summary(self, tt: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(tt)) if len(tt) else float("nan"),
            "std": float(np.std(tt)) if len(tt) else float("nan"),
            "p50": float(np.percentile(tt, 50)) if len(tt) else float("nan"),
            "p90": float(np.percentile(tt, 90)) if len(tt) else float("nan"),
            "p95": float(np.percentile(tt, 95)) if len(tt) else float("nan"),
            "p99": float(np.percentile(tt, 99)) if len(tt) else float("nan"),
        }

    def run_streaming(self, batch_size: int = 500):
        """Generator: liefert nach jedem Batch einen Update-Dict."""
        self.reset()
        times = self.inc_gen.sample_incident_times()
        locs = self.inc_gen.sample_incident_locations(len(times))
        n = len(times)

        travel_times_min = np.full(n, np.nan, dtype=float)
        recent: List[Tuple[float, float]] = []
        t0 = time.time()

        for idx, (t_inc, (lat, lon)) in enumerate(zip(times, locs)):
            target_node = self.router.nearest_node(lat, lon)
            candidates = self._candidate_units(lat, lon, self.p.k_candidates)

            best_eta = float('inf'); best_unit = None; best_t_travel_min = None
            for u in candidates:
                depart = max(t_inc, u.free_at)
                tt_sec = self.router.tt_seconds(u.node, target_node)
                if not math.isfinite(tt_sec):
                    continue
                eta = depart + tt_sec/60.0
                if eta < best_eta:
                    best_eta = eta; best_unit = u; best_t_travel_min = tt_sec/60.0

            if best_unit is None:
                # Fallback: prüfe alle
                for u in self.ambulances:
                    depart = max(t_inc, u.free_at)
                    tt_sec = self.router.tt_seconds(u.node, target_node)
                    if not math.isfinite(tt_sec):
                        continue
                    eta = depart + tt_sec/60.0
                    if eta < best_eta:
                        best_eta = eta; best_unit = u; best_t_travel_min = tt_sec/60.0

            # Metriken + Status pflegen
            if best_unit is not None:
                travel_times_min[idx] = best_t_travel_min
                arrival = max(t_inc, best_unit.free_at) + best_t_travel_min
                best_unit.node = target_node
                best_unit.latlon = (lat, lon)
                best_unit.free_at = arrival + self.p.service_time_min

            recent.append((lat, lon))
            if len(recent) > 500:
                recent.pop(0)

            # Batch-Update
            if ((idx + 1) % batch_size == 0) or (idx + 1 == n):
                valid = np.isfinite(travel_times_min[:idx+1])
                tt = travel_times_min[:idx+1][valid]
                elapsed = time.time() - t0
                ips = (idx + 1) / elapsed if elapsed > 0 else 0.0
                yield {
                    "processed": idx + 1,
                    "total": n,
                    "elapsed_s": elapsed,
                    "incidents_per_s": ips,
                    "recent": np.array(recent),
                    **self._summary(tt),
                }

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="EMS ABM Hamburg – Live Dashboard", layout="wide")

st.title("🚑 EMS‑ABM Hamburg – Live‑Dashboard (OSM‑Routing)")
st.caption("Batch‑basierte Echtzeit‑Kennzahlen der Fahrtdauer zum Einsatzort.")

with st.sidebar:
    st.header("⚙️ Parameter")
    place = st.text_input("OSM Place", "Hamburg, Germany")
    graphml = st.text_input("GraphML Cache", "hamburg_drive.graphml")
    n_inc = st.number_input("Einsätze gesamt", min_value=1000, max_value=200000, value=50000, step=5000)
    n_units = st.number_input("Anzahl RTW", min_value=5, max_value=200, value=30, step=5)
    service_min = st.number_input("Servicezeit (min)", min_value=0.0, max_value=120.0, value=20.0, step=5.0)
    k_cand = st.number_input("k‑Kandidaten", min_value=1, max_value=30, value=5, step=1)
    batch = st.number_input("Batch‑Größe (Updates)", min_value=50, max_value=5000, value=500, step=50)
    seed = st.number_input("Seed", min_value=0, max_value=10**9, value=42, step=1)
    start = st.button("▶️ Simulation starten")

# Live‑Placeholders
cols = st.columns(6)
m_proc = cols[0].empty()
m_mean = cols[1].empty()
m_med  = cols[2].empty()
m_p90  = cols[3].empty()
m_p95  = cols[4].empty()
m_p99  = cols[5].empty()

map_placeholder = st.empty()
progress = st.progress(0.0)
chart_placeholder = st.empty()
log_placeholder = st.empty()

if start:
    # Model aufbauen
    params = SimParams(
        place_name=place, graphml_path=graphml,
        n_incidents=int(n_inc), n_units=int(n_units),
        service_time_min=float(service_min), k_candidates=int(k_cand),
        seed=int(seed),
        speed_fallback_kph=30.0,
        hwy_speeds=None,
    )

    model = EMSModel(params)

    # Live‑Chart: Verlauf P50
    chart_df = pd.DataFrame(columns=["processed","p50","p90","mean"])
    chart = None

    # Streaming‑Run
    for upd in model.run_streaming(batch_size=int(batch)):
        processed = upd["processed"]; total = upd["total"]
        pct = processed / total

        # Metrics
        m_proc.metric("Verarbeitet", f"{processed} / {total}", f"{upd['incidents_per_s']:.1f}/s")
        m_mean.metric("Mean (min)", f"{upd['mean']:.2f}")
        m_med.metric("Median (min)", f"{upd['p50']:.2f}")
        m_p90.metric("P90 (min)", f"{upd['p90']:.2f}")
        m_p95.metric("P95 (min)", f"{upd['p95']:.2f}")
        m_p99.metric("P99 (min)", f"{upd['p99']:.2f}")

        # Fortschritt
        progress.progress(min(pct, 1.0))

        if len(upd["recent"]):
            df_map = pd.DataFrame(upd["recent"], columns=["lat","lon"])
            map_placeholder.map(df_map)

        # Chart aktualisieren
        chart_df.loc[len(chart_df)] = [processed, upd["p50"], upd["p90"], upd["mean"]]
        if chart is None:
            chart = chart_placeholder.line_chart(chart_df.set_index("processed"))
        else:
            chart.add_rows(chart_df.set_index("processed").iloc[-1:])

        # Log
        log_placeholder.info(
            f"Batch Update – processed={processed}/{total} "
            f"elapsed={upd['elapsed_s']:.1f}s ips={upd['incidents_per_s']:.1f}"
        )

    st.success("Simulation abgeschlossen.")

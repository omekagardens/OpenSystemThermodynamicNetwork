"""
det_gtp_server.py

FastAPI server for DET-GTP.

Implements:
- 4.1 DET Brain (persistent, identity-bearing component)
- 4.2 Bodies (replaceable clients; do not learn/persist)
- 4.3 Teacher LLM (developmental scaffold; optional feedback channel)
- 6.1 Semantic Protocol Layer (Semantic Anchors + Anchor Packet)

This server hosts one or more ContinuousDETBrain instances (keyed by brain_id / soul_id),
persists them to disk, and emits body-agnostic SemanticPackets for Swift or other bodies.

Run:
  python det_gtp_server.py

Then from Swift:
  GET  http://127.0.0.1:8717/v1/health
  POST http://127.0.0.1:8717/v1/brains/<id>/ensure
  POST http://127.0.0.1:8717/v1/brains/<id>/observe
  POST http://127.0.0.1:8717/v1/brains/<id>/tick
  GET  http://127.0.0.1:8717/v1/brains/<id>/packet?top_k=12
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import argparse
import shlex
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

# --- Import your existing brain (kept separate for easy patching/testing) ---
# Expect this file to live near det_brain_continuous.py OR be importable on PYTHONPATH.
try:
    from det_brain_continuous import ContinuousDETBrain  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import ContinuousDETBrain from det_brain_continuous.py. "
        "Place det_gtp_server.py in the same folder or fix your PYTHONPATH."
    ) from e


# =============================================================================
# Semantic Protocol Layer (6.1)
# =============================================================================

SEMANTIC_PROTOCOL_VERSION = "0.1.0"


def _stable_anchor_id(namespace: str, label: str) -> str:
    """
    Body-independent stable id.
    Deterministic: same (namespace,label) => same id across runs/machines.
    """
    h = hashlib.sha256(f"{namespace}::{label}".encode("utf-8")).hexdigest()
    return f"sa_{h[:24]}"


@dataclass(frozen=True)
class SemanticAnchor:
    """
    Minimal, body-independent handle for meaning (6.1).
    Not tokens-as-cognition; just a stable checksum-like identity + small tags.
    """
    id: str
    label: str
    tags: List[str]
    polarity: float  # e.g. sign of pressure, normalized-ish
    salience: float  # e.g. |pressure| or delta-pressure


@dataclass(frozen=True)
class SemanticRelation:
    """
    Relation between anchors.
    NOTE: We emit anchor IDs, not internal graph node identifiers.
    """
    src: str
    dst: str
    kind: str          # e.g. "assoc"
    weight: float      # e.g. conductance proxy


@dataclass(frozen=True)
class SemanticPacket:
    """
    The Anchor Packet: anchors + relations + constraints + debug metadata.
    Bodies consume this; they do not see raw internal node IDs or full topology.
    """
    protocol_version: str
    brain_id: str
    mode: str
    anchors: List[SemanticAnchor]
    relations: List[SemanticRelation]
    constraints: Dict[str, Any]
    meta: Dict[str, Any]


# =============================================================================
# Requests / Responses
# =============================================================================

class EnsureResponse(BaseModel):
    brain_id: str
    created: bool
    loaded_from_disk: bool


class ObserveRequest(BaseModel):
    text: str = Field(..., description="External stimulus (user text, sensor event, etc.)")
    strength: float = Field(5.0, ge=0.0, le=1000.0)
    source: str = Field("body", description="body|teacher|system")
    note: Optional[str] = None


class ObserveResponse(BaseModel):
    brain_id: str
    observed: List[str]
    ticks: int


class TickRequest(BaseModel):
    steps: int = Field(1, ge=1, le=10_000)


class TickResponse(BaseModel):
    brain_id: str
    ticks: int


class TeacherFeedbackRequest(BaseModel):
    """
    Teacher scaffold feedback (4.3):
    - Adjust relative importance without becoming the brain.
    - Keep changes inspectable & reversible.
    """
    kind: str = Field(..., description="edge_boost|edge_dampen|tag_anchor|note")
    a_label: Optional[str] = None
    b_label: Optional[str] = None
    amount: float = Field(0.1, ge=0.0, le=10.0)
    tag: Optional[str] = None
    note: Optional[str] = None


class StatsResponse(BaseModel):
    brain_id: str
    ticks: int
    num_nodes: int
    num_edges: int
    top_nodes: List[Tuple[str, float]]


class PacketResponse(BaseModel):
    packet: Dict[str, Any]


# =============================================================================
# Brain Store (4.1 persistence)
# =============================================================================

class BrainStore:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self._brains: Dict[str, ContinuousDETBrain] = {}
        self._created_at: Dict[str, float] = {}
        self._mode: Dict[str, str] = {}  # Phase/mode label, e.g. "nursery"|"shared"|"mature"
        self._audit_log: Dict[str, List[Dict[str, Any]]] = {}

    def _state_path(self, brain_id: str) -> str:
        safe = "".join(ch for ch in brain_id if ch.isalnum() or ch in ("-", "_")).strip()
        if not safe:
            raise ValueError("Invalid brain_id")
        return os.path.join(self.state_dir, f"{safe}.json")

    def ensure(self, brain_id: str) -> Tuple[ContinuousDETBrain, bool, bool]:
        """
        Return (brain, created, loaded_from_disk).
        """
        if brain_id in self._brains:
            return self._brains[brain_id], False, False

        path = self._state_path(brain_id)
        if os.path.exists(path):
            brain = ContinuousDETBrain.load_state(path)
            created = False
            loaded = True
        else:
            brain = ContinuousDETBrain()
            created = True
            loaded = False

        self._brains[brain_id] = brain
        self._created_at[brain_id] = time.time()
        self._mode.setdefault(brain_id, "nursery")  # default developmental mode
        self._audit_log.setdefault(brain_id, [])

        return brain, created, loaded

    def get(self, brain_id: str) -> ContinuousDETBrain:
        brain, _, _ = self.ensure(brain_id)
        return brain

    def save(self, brain_id: str) -> None:
        brain = self.get(brain_id)
        path = self._state_path(brain_id)
        brain.save_state(path)

    def mode(self, brain_id: str) -> str:
        self.ensure(brain_id)
        return self._mode.get(brain_id, "nursery")

    def set_mode(self, brain_id: str, mode: str) -> None:
        self.ensure(brain_id)
        self._mode[brain_id] = mode
        self._audit_log[brain_id].append({
            "t": time.time(),
            "type": "mode_set",
            "mode": mode,
        })

    def log(self, brain_id: str, entry: Dict[str, Any]) -> None:
        self.ensure(brain_id)
        self._audit_log[brain_id].append(entry)

    def get_log(self, brain_id: str, tail: int = 200) -> List[Dict[str, Any]]:
        self.ensure(brain_id)
        return self._audit_log[brain_id][-tail:]


# =============================================================================
# Semantic Packet construction (6.1)
# =============================================================================

def _build_packet(
    brain_id: str,
    brain: ContinuousDETBrain,
    mode: str,
    *,
    top_k: int,
    query: Optional[str] = None
) -> SemanticPacket:
    """
    Emit a body-agnostic packet:
      anchors = salient concepts
      relations = lightweight edges among those anchors (bounded)
      constraints = budgets & hints for bodies
    """

    namespace = "det_gtp"

    # If a query is provided, we use delta-pressure contextual readout.
    if query and query.strip():
        headline, ctx = brain.read_contextual_thought_structured(query, top_k=max(1, top_k - 1))
        chosen_labels: List[str] = []
        if headline:
            chosen_labels.append(headline)
        chosen_labels.extend(ctx)
        # Fallback: if nothing chosen, use top thoughts
        if not chosen_labels:
            chosen_labels = [n for n, _p in brain.read_thoughts(top_k=top_k)]
    else:
        chosen_labels = [n for n, _p in brain.read_thoughts(top_k=top_k)]

    # Deduplicate while preserving order
    seen = set()
    labels: List[str] = []
    for lab in chosen_labels:
        if lab not in seen:
            labels.append(lab)
            seen.add(lab)
        if len(labels) >= top_k:
            break

    anchors: List[SemanticAnchor] = []
    label_to_anchor_id: Dict[str, str] = {}

    # Use current pressures for salience/polarity (simple, inspectable)
    for lab in labels:
        p = float(brain.node_pressure.get(lab, 0.0))
        aid = _stable_anchor_id(namespace, lab)
        label_to_anchor_id[lab] = aid
        anchors.append(SemanticAnchor(
            id=aid,
            label=lab,
            tags=[],
            polarity=1.0 if p >= 0 else -1.0,
            salience=abs(p),
        ))

    # Relations: only among the chosen set, bounded.
    # We do NOT emit full topology. Only a small induced subgraph slice.
    relations: List[SemanticRelation] = []
    max_rel = min(80, top_k * (top_k - 1))  # bounded

    # Access underlying graph edges; only keep those linking chosen labels.
    chosen_set = set(labels)
    for (u, v) in getattr(brain.graph, "edges", lambda: [])():
        if u in chosen_set and v in chosen_set:
            key = tuple(sorted((u, v)))
            sigma = float(brain.edge_conductance.get(key, 1.0))
            relations.append(SemanticRelation(
                src=label_to_anchor_id[u],
                dst=label_to_anchor_id[v],
                kind="assoc",
                weight=sigma,
            ))
            if len(relations) >= max_rel:
                break

    constraints = {
        "top_k": top_k,
        "max_relations": max_rel,
        # Expose hard/soft budgets the *body* can respect while rendering.
        # (The real enforcement is in the brain dynamics itself.)
        "budgets": {
            "active_nodes_max_hint": top_k,
            "relations_max_hint": max_rel,
        },
    }

    meta = {
        "ticks": int(brain.tick_index),
        "num_nodes": int(brain.graph.number_of_nodes()),
        "num_edges": int(brain.graph.number_of_edges()),
        "query": query or "",
        "mode": mode,
        "created_at": None,
    }

    return SemanticPacket(
        protocol_version=SEMANTIC_PROTOCOL_VERSION,
        brain_id=brain_id,
        mode=mode,
        anchors=anchors,
        relations=relations,
        constraints=constraints,
        meta=meta,
    )



# =============================================================================
# Offline REPL (server not running)
# =============================================================================

def _brain_state_path(state_dir: str, brain_id: str) -> str:
    safe = "".join(ch for ch in brain_id if ch.isalnum() or ch in ("-", "_")).strip()
    if not safe:
        raise ValueError("Invalid brain_id")
    return os.path.join(state_dir, f"{safe}.json")


def _load_or_create_brain_for_repl(state_dir: str, brain_id: str) -> ContinuousDETBrain:
    path = _brain_state_path(state_dir, brain_id)
    if os.path.exists(path):
        return ContinuousDETBrain.load_state(path)
    return ContinuousDETBrain()


def _save_brain_for_repl(state_dir: str, brain_id: str, brain: ContinuousDETBrain) -> None:
    os.makedirs(state_dir, exist_ok=True)
    path = _brain_state_path(state_dir, brain_id)
    brain.save_state(path)


def run_repl(state_dir: str, brain_id: str) -> None:
    """Interactive shell for directly manipulating a brain state on disk.

    Intended usage: run this when the FastAPI server is NOT running.
    Operates on the same on-disk state files as the server.

    Commands:
      /help
      /load <brain_id>
      /save
      /stats
      /packet [top_k] [query...]
      /tick [steps]
      /observe <text...>
      /train <json_path> [epochs]
      /mode <nursery|shared|mature|custom>
      /exit

    Bare text (not starting with '/') is treated as: observe(text) then tick(5).
    """

    os.makedirs(state_dir, exist_ok=True)

    current_id = brain_id
    brain = _load_or_create_brain_for_repl(state_dir, current_id)
    tick_default = 5

    def _print_banner() -> None:
        print(">>> DET-GTP Offline REPL <<<")
        print(f"state_dir: {os.path.abspath(state_dir)}")
        print(f"brain_id:  {current_id}")
        print("Type /help for commands. Ctrl+C or /exit to quit.")

    def _print_help() -> None:
        print(
            "\n".join([
                "Commands:",
                "  /help",
                "  /load <brain_id>                 (switch brains)",
                "  /save                            (persist current brain)",
                "  /stats                           (nodes/edges/top pressures)",
                "  /packet [top_k] [query...]        (semantic packet preview)",
                "  /tick [steps]                     (advance dynamics)",
                "  /observe <text...>                (observe stimulus without ticking)",
                "  /train <json_path> [epochs]       (train from JSON, if supported)",
                "  /mode <nursery|shared|mature|...> (set a mode label in state file meta, if supported)",
                "  /exit",
                "",
                "Bare text:",
                f"  observe(text) then tick({tick_default})",
                ""
            ])
        )

    # Optional: lightweight mode label stored alongside brain state if the brain supports metadata.
    mode_label = "nursery"

    _print_banner()

    try:
        while True:
            try:
                line = input("det_gtp> ").strip()
            except EOFError:
                print()
                break

            if not line:
                continue

            # Slash commands
            if line.startswith("/"):
                parts = shlex.split(line)
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in ("/exit", "/quit"):
                    break

                if cmd == "/help":
                    _print_help()
                    continue

                if cmd == "/load":
                    if not args:
                        print("Usage: /load <brain_id>")
                        continue
                    # Save current before switching
                    _save_brain_for_repl(state_dir, current_id, brain)
                    current_id = args[0]
                    brain = _load_or_create_brain_for_repl(state_dir, current_id)
                    print(f"Loaded brain_id={current_id} (ticks={getattr(brain, 'tick_index', 0)})")
                    continue

                if cmd == "/save":
                    _save_brain_for_repl(state_dir, current_id, brain)
                    print(f"Saved: {_brain_state_path(state_dir, current_id)}")
                    continue

                if cmd == "/stats":
                    if hasattr(brain, "get_stats"):
                        s = brain.get_stats(top_k=10)
                        print(json.dumps(s, indent=2))
                    else:
                        # Fallback stats
                        num_nodes = int(brain.graph.number_of_nodes())
                        num_edges = int(brain.graph.number_of_edges())
                        tops = sorted(getattr(brain, "node_pressure", {}).items(), key=lambda kv: kv[1], reverse=True)[:10]
                        print(json.dumps({
                            "ticks": int(getattr(brain, "tick_index", 0)),
                            "num_nodes": num_nodes,
                            "num_edges": num_edges,
                            "top_nodes": [(n, float(p)) for n, p in tops],
                        }, indent=2))
                    continue

                if cmd == "/tick":
                    steps = int(args[0]) if args else tick_default
                    if hasattr(brain, "tick"):
                        brain.tick(steps=steps)
                        print(f"ticks={int(getattr(brain, 'tick_index', 0))}")
                    else:
                        print("Brain has no tick()")
                    continue

                if cmd == "/observe":
                    if not args:
                        print("Usage: /observe <text...>")
                        continue
                    text = " ".join(args)
                    if hasattr(brain, "observe_text"):
                        obs = brain.observe_text(text, strength=5.0)
                        print(f"observed={len(obs)}")
                    else:
                        print("Brain has no observe_text()")
                    continue

                if cmd == "/packet":
                    # /packet [top_k] [query...]
                    top_k = 12
                    query = None
                    if args:
                        # if first arg is int, treat as top_k
                        try:
                            top_k = int(args[0])
                            rest = args[1:]
                        except ValueError:
                            rest = args
                        if rest:
                            query = " ".join(rest)
                    pkt = _build_packet(current_id, brain, mode_label, top_k=top_k, query=query)
                    pkt_dict = {
                        "protocol_version": pkt.protocol_version,
                        "brain_id": pkt.brain_id,
                        "mode": pkt.mode,
                        "anchors": [asdict(a) for a in pkt.anchors],
                        "relations": [asdict(r) for r in pkt.relations],
                        "constraints": pkt.constraints,
                        "meta": pkt.meta,
                    }
                    print(json.dumps(pkt_dict, indent=2))
                    continue

                if cmd == "/train":
                    if not args:
                        print("Usage: /train <json_path> [epochs]")
                        continue
                    json_path = Path(args[0]).expanduser()
                    epochs = int(args[1]) if len(args) >= 2 else 1
                    if not json_path.exists():
                        print(f"File not found: {json_path}")
                        continue

                    # Support either train_from_json(path, epochs=...) or train_from_json(data, epochs=...)
                    if hasattr(brain, "train_from_json"):
                        try:
                            # Prefer path-based signature
                            brain.train_from_json(str(json_path), epochs=epochs)  # type: ignore
                            print(f"Trained from {json_path} (epochs={epochs})")
                        except TypeError:
                            # Fallback: load JSON into memory
                            data = json.loads(json_path.read_text(encoding="utf-8"))
                            brain.train_from_json(data, epochs=epochs)  # type: ignore
                            print(f"Trained from {json_path} (epochs={epochs})")
                    else:
                        print("Brain does not support /train yet (no train_from_json method)")
                    continue

                if cmd == "/mode":
                    if not args:
                        print("Usage: /mode <nursery|shared|mature|custom>")
                        continue
                    mode_label = args[0]
                    print(f"mode={mode_label}")
                    continue

                print(f"Unknown command: {cmd}. Try /help")
                continue

            # Bare text: observe + tick
            text = line
            if hasattr(brain, "observe_text"):
                obs = brain.observe_text(text, strength=5.0)
                print(f"observed={len(obs)}", end="  ")
            else:
                print("observed=?", end="  ")

            if hasattr(brain, "tick"):
                brain.tick(steps=tick_default)
                print(f"ticks={int(getattr(brain, 'tick_index', 0))}")
            else:
                print("ticks=?")

            # Persist after each turn (keeps parity with server behavior)
            _save_brain_for_repl(state_dir, current_id, brain)

    except KeyboardInterrupt:
        print("\nShutting down.")

    # Final save
    _save_brain_for_repl(state_dir, current_id, brain)
    print(f"Saved: {_brain_state_path(state_dir, current_id)}")


# =============================================================================
# FastAPI App
# =============================================================================

APP_HOST = os.environ.get("DET_GTP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("DET_GTP_PORT", "8717"))
STATE_DIR = os.environ.get("DET_GTP_STATE_DIR", "./det_gtp_state")

store = BrainStore(state_dir=STATE_DIR)
app = FastAPI(title="DET-GTP Server", version=SEMANTIC_PROTOCOL_VERSION)


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "protocol_version": SEMANTIC_PROTOCOL_VERSION,
        "state_dir": os.path.abspath(STATE_DIR),
    }


@app.post("/v1/brains/{brain_id}/ensure", response_model=EnsureResponse)
def ensure_brain(brain_id: str) -> EnsureResponse:
    _brain, created, loaded = store.ensure(brain_id)
    return EnsureResponse(brain_id=brain_id, created=created, loaded_from_disk=loaded)


@app.post("/v1/brains/{brain_id}/observe", response_model=ObserveResponse)
def observe(brain_id: str, req: ObserveRequest) -> ObserveResponse:
    brain = store.get(brain_id)

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    observed = brain.observe_text(req.text, strength=req.strength)

    store.log(brain_id, {
        "t": time.time(),
        "type": "observe",
        "source": req.source,
        "strength": req.strength,
        "observed_count": len(observed),
        "note": req.note,
    })

    # Persist after observation (simple + safe)
    store.save(brain_id)

    return ObserveResponse(brain_id=brain_id, observed=observed, ticks=int(brain.tick_index))


@app.post("/v1/brains/{brain_id}/tick", response_model=TickResponse)
def tick(brain_id: str, req: TickRequest) -> TickResponse:
    brain = store.get(brain_id)
    brain.tick(steps=req.steps)

    store.log(brain_id, {
        "t": time.time(),
        "type": "tick",
        "steps": req.steps,
    })

    store.save(brain_id)
    return TickResponse(brain_id=brain_id, ticks=int(brain.tick_index))


@app.get("/v1/brains/{brain_id}/stats", response_model=StatsResponse)
def stats(brain_id: str) -> StatsResponse:
    brain = store.get(brain_id)
    s = brain.get_stats(top_k=10)
    return StatsResponse(
        brain_id=brain_id,
        ticks=int(s["ticks"]),
        num_nodes=int(s["num_nodes"]),
        num_edges=int(s["num_edges"]),
        top_nodes=[(str(n), float(p)) for n, p in s["top_nodes"]],
    )


@app.get("/v1/brains/{brain_id}/packet", response_model=PacketResponse)
def packet(
    brain_id: str,
    top_k: int = Query(12, ge=1, le=200),
    query: Optional[str] = Query(None, description="Optional query to produce contextual delta-pressure packet"),
) -> PacketResponse:
    brain = store.get(brain_id)
    mode = store.mode(brain_id)
    pkt = _build_packet(brain_id, brain, mode, top_k=top_k, query=query)

    # Emit as plain JSON dict
    pkt_dict = {
        "protocol_version": pkt.protocol_version,
        "brain_id": pkt.brain_id,
        "mode": pkt.mode,
        "anchors": [asdict(a) for a in pkt.anchors],
        "relations": [asdict(r) for r in pkt.relations],
        "constraints": pkt.constraints,
        "meta": pkt.meta,
    }

    return PacketResponse(packet=pkt_dict)


@app.post("/v1/brains/{brain_id}/mode")
def set_mode(brain_id: str, mode: str = Query(..., description="nursery|shared|mature|custom")) -> Dict[str, Any]:
    store.set_mode(brain_id, mode)
    store.save(brain_id)
    return {"brain_id": brain_id, "mode": mode}


@app.get("/v1/brains/{brain_id}/log")
def get_log(brain_id: str, tail: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    return {"brain_id": brain_id, "log": store.get_log(brain_id, tail=tail)}


# -----------------------------------------------------------------------------
# Teacher scaffold hook (4.3) — optional and inspectable
# -----------------------------------------------------------------------------

@app.post("/v1/brains/{brain_id}/teacher_feedback")
def teacher_feedback(brain_id: str, req: TeacherFeedbackRequest) -> Dict[str, Any]:
    brain = store.get(brain_id)

    # Keep it minimal + reversible:
    # - "edge_boost"/"edge_dampen" adjusts conductance for an association
    # - "tag_anchor" adds a tag (server-side only for now)
    # - "note" just logs
    action = req.kind.strip().lower()

    if action in ("edge_boost", "edge_dampen"):
        if not req.a_label or not req.b_label:
            raise HTTPException(status_code=400, detail="a_label and b_label required")

        a = req.a_label.lower().strip()
        b = req.b_label.lower().strip()
        if not a or not b:
            raise HTTPException(status_code=400, detail="empty labels")

        # Ensure concepts exist and are connected (or connect lightly)
        brain.add_concept(a)
        brain.add_concept(b)
        if not brain.graph.has_edge(a, b):
            brain.associate(a, b, weight=0.05)

        key = tuple(sorted((a, b)))
        sigma = float(brain.edge_conductance.get(key, 1.0))

        if action == "edge_boost":
            sigma = sigma + float(req.amount)
        else:
            sigma = max(1e-4, sigma - float(req.amount))

        brain.edge_conductance[key] = sigma

        store.log(brain_id, {
            "t": time.time(),
            "type": "teacher_feedback",
            "kind": action,
            "a_label": a,
            "b_label": b,
            "amount": req.amount,
            "new_sigma": sigma,
            "note": req.note,
        })

    elif action == "tag_anchor":
        # Tags are protocol-level, but your ContinuousDETBrain doesn't track them yet.
        # We log tags for now (inspectable), and you can later store them in brain state.
        if not req.a_label or not req.tag:
            raise HTTPException(status_code=400, detail="a_label and tag required")

        store.log(brain_id, {
            "t": time.time(),
            "type": "teacher_feedback",
            "kind": action,
            "a_label": req.a_label.lower().strip(),
            "tag": req.tag,
            "note": req.note,
        })

    elif action == "note":
        store.log(brain_id, {
            "t": time.time(),
            "type": "teacher_feedback",
            "kind": action,
            "note": req.note,
        })

    else:
        raise HTTPException(status_code=400, detail=f"unknown kind: {req.kind}")

    store.save(brain_id)
    return {"ok": True}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DET-GTP server (FastAPI) and offline REPL")
    parser.add_argument("--host", default=APP_HOST)
    parser.add_argument("--port", type=int, default=APP_PORT)
    parser.add_argument("--state-dir", default=STATE_DIR)

    parser.add_argument("--repl", action="store_true", help="Run offline REPL instead of server")
    parser.add_argument("--brain-id", default="default", help="Brain id for offline REPL")

    args = parser.parse_args()

    if args.repl:
        run_repl(state_dir=args.state_dir, brain_id=args.brain_id)
    else:
        # Server mode
        uvicorn.run(app, host=args.host, port=args.port)
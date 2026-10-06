"""
The local chat server. It uses Python's built-in http.server: no web framework, no
internet, no AI service.

    python -m chat.server                 # start the newest brain, open http://localhost:8000
    python -m chat.server --stage 1       # talk to an older brain instead (1, 1.5, 1.5b, 1.5b2, 1.5b3, 1.5b4, 1.5b4b, 1.5b4c)
    python -m chat.server --port 8001     # use another port
    python -m chat.server --no-browser    # don't open the browser automatically
    python -m chat.server --no-memory     # chat without long-term memory (nothing is saved)
    python -m chat.server --no-learning   # chat without learning from experience (no knowledge is kept)
    python -m chat.server --no-reasoning  # without the world model and reasoning (Stage 2c): recall as in 2b
    python -m chat.server --no-self       # without the self-model and its event log (nothing about the system is recorded)
    python -m chat.server --no-state      # without the internal state (it also needs the event log: off with --no-self)

    Browser -> this server -> ChatEngine -> [long-term memory] -> OUR neural network -> reply -> browser

Long-term memory and learning are ON by default: saved in state/memory.db and state/knowledge.db on this
Mac, they survive closing the chat and restarting the computer (see memory/ and learning/).

It listens on 127.0.0.1 only, which means only this Mac can connect to it.

Addresses it answers:
    GET  /              the chat page
    GET  /static/...    the page's style sheet and script
    GET  /evaluation    the evaluation dashboard (made by python -m brain.evaluate)
    GET  /api/info      facts about the loaded model
    GET  /api/history   the current conversation
    POST /api/chat      {"message": "..."}  ->  {"reply": "...", "details": {...}}
    POST /api/clear     clear the conversation on screen (long-term memory is kept)

    GET  /api/memories              what is stored: memories, and conversation-history counts
    GET  /api/memories/why?id=N     why memory N exists, where it came from, how it was used
    POST /api/memories/delete       {"id": N}           delete one memory
    POST /api/memories/important    {"id": N, "important": true}
    POST /api/memories/clear        {"confirm": true}   delete every memory
    POST /api/history/clear         {"confirm": true}   delete the stored conversation history

    GET  /api/knowledge             what was learned: grouped facts, connections, the learning experience log
    GET  /api/knowledge/why?id=N    where fact N came from, its history (earlier values)
    POST /api/knowledge/remove      {"id": N}           remove one learned fact (and its history)
    POST /api/knowledge/clear       {"confirm": true}   remove everything learned

    GET  /api/world                 the world model: entities, properties, relationships, history, last reasoning
    GET  /api/world/reason?q=...    run a question through interpretation, retrieval and reasoning only
                                    (no brain, nothing learned or stored)

    GET  /api/self                  the self-model: current state, capabilities, experience, uncertainty,
                                    history, environment (computed from the event log and the stores)
    GET  /api/self/ask?q=...        a question about recorded state -> its report with evidence (no brain)
    GET  /api/self/events           the newest events of the event log
    POST /api/self/clear            {"confirm": true}   delete the event log (the current run's start is kept)

    GET  /api/state                 the internal state: current numbers, definitions, history with causes and
                                    evidence, expectations, memory tags, open gaps (never shown to the brain)
    POST /api/state/reset           {"what": "baselines" | "everything", "confirm": true}
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote_plus

from brain.generate import (STAGE1_MODEL, STAGE15_MODEL, STAGE15B2_MODEL, STAGE15B3_MODEL, STAGE15B4_MODEL,
                            STAGE15B4B_MODEL, STAGE15B4C_MODEL, STAGE15B4D_MODEL, STAGE15B_MODEL, ModelNotFoundError,
                            load_brain, newest_model)
from chat.chat_engine import ChatEngine
from learning.knowledge import DEFAULT_DB as KNOWLEDGE_DB
from learning.system import LearningSystem
from memory.store import DEFAULT_DB
from memory.system import MemorySystem
from internal_state.store import DEFAULT_DB as STATE_DB
from internal_state.system import InternalState
from self_model.events import DEFAULT_DB as EVENTS_DB
from self_model.system import SelfSystem
from world.system import WorldSystem

HOST = "127.0.0.1"          # this Mac only; never exposed to the network
DEFAULT_PORT = 8000
STATIC_DIR = Path(__file__).resolve().parent / "static"
PAGES = {                    # the only files the server will ever send
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 10_000


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "OurAI/1"

    # ------------------------------------------------------------------ helpers
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("The message is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def log_message(self, fmt, *args):
        pass                  # keep the Terminal quiet; errors are printed separately

    # ---------------------------------------------------------------- GET / POST
    def do_GET(self):
        path = self.path.split("?")[0]
        engine, lock = self.server.engine, self.server.lock
        if path in PAGES:
            filename, content_type = PAGES[path]
            body = (STATIC_DIR / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/evaluation":                  # the dashboard made by python -m brain.evaluate
            report = Path(__file__).resolve().parent.parent / "models" / "evaluation.html"
            if not report.exists():
                self.send_json(404, {"error": "No evaluation yet. Run: python -m brain.evaluate"})
                return
            body = report.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/info":
            self.send_json(200, {**engine.info(), "memory": engine.memory is not None,
                                 "learning": engine.learning is not None, "reasoning": engine.world is not None,
                                 "self_model": engine.self_model is not None,
                                 "internal_state": engine.internal_state is not None})
        elif path == "/api/state":
            if engine.internal_state is None:
                self.send_json(200, {"enabled": False})
                return
            with lock:
                self.send_json(200, engine.internal_state.view())
        elif path in ("/api/self", "/api/self/ask", "/api/self/events"):
            self.self_view(path, engine, lock)
        elif path == "/api/world":
            if engine.world is None:
                self.send_json(200, {"enabled": False})
                return
            with lock:
                self.send_json(200, {"enabled": True, **engine.world.view()})
        elif path == "/api/world/reason":
            raw = self.path.partition("?")[2]
            query = dict(part.split("=", 1) for part in raw.split("&") if "=" in part)
            question = unquote_plus(query.get("q", "")).strip()[:500]
            if engine.world is None or not question:
                self.send_json(400, {"error": "Reasoning is off, or there is no question."})
                return
            with lock:
                self.send_json(200, engine.world.explain(question, engine.answer_format))
        elif path == "/api/memories":
            if engine.memory is None:
                self.send_json(200, {"enabled": False, "memories": []})
                return
            with lock:
                self.send_json(200, {"enabled": True, "memories": engine.memory.memories(),
                                     "history": engine.memory.history_counts(),
                                     "database": str(engine.memory.path.name)})
        elif path == "/api/knowledge":
            if engine.learning is None:
                self.send_json(200, {"enabled": False})
                return
            with lock:
                self.send_json(200, {"enabled": True, **engine.learning.view()})
        elif path == "/api/knowledge/why":
            query = dict(part.split("=", 1) for part in self.path.partition("?")[2].split("&") if "=" in part)
            with lock:
                found = engine.learning.why(int(query["id"])) if engine.learning and \
                    query.get("id", "").isdigit() else None
            if found:
                self.send_json(200, found)
            else:
                self.send_json(404, {"error": "No such fact."})
        elif path == "/api/memories/why":
            query = dict(part.split("=", 1) for part in self.path.partition("?")[2].split("&") if "=" in part)
            with lock:
                found = engine.memory.why(int(query.get("id", 0))) if engine.memory and \
                    query.get("id", "").isdigit() else None
            if found:
                self.send_json(200, found)
            else:
                self.send_json(404, {"error": "No such memory."})
        elif path == "/api/history":
            with lock:
                self.send_json(200, {"history": engine.public_history()})
        else:
            self.send_json(404, {"error": "Not found."})

    def do_POST(self):
        path = self.path.split("?")[0]
        engine, lock = self.server.engine, self.server.lock
        try:
            data = self.read_json()
        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {"error": "The request could not be read."})
            return
        if path == "/api/chat":
            try:
                with lock:            # one reply at a time
                    result = engine.reply(data.get("message"))
            except ValueError as e:
                self.send_json(400, {"error": str(e)})
                return
            except Exception as e:     # show the problem instead of hanging the page
                print(f"  error while generating: {type(e).__name__}: {e}", file=sys.stderr)
                self.send_json(500, {"error": f"The model hit an error: {type(e).__name__}. "
                                              "See the Terminal for details."})
                return
            self.send_json(200, result)
        elif path == "/api/clear":
            with lock:
                engine.clear()
                if engine.self_model is not None:
                    engine.self_model.control("cleared the conversation on screen")
            self.send_json(200, {"ok": True})
        elif path == "/api/state/reset":
            what = data.get("what")
            if engine.internal_state is None or data.get("confirm") is not True or what not in ("baselines", "everything"):
                self.send_json(400, {"error": "The internal state is off, or the request was not understood."})
                return
            with lock:
                engine.internal_state.reset(what)
                engine.self_model.control("reset the internal state's " + ("baselines" if what == "baselines"
                                                                           else "record (everything)"))
            self.send_json(200, {"ok": True})
        elif path == "/api/self/clear":
            if engine.self_model is None or data.get("confirm") is not True:
                self.send_json(400, {"error": "The self-model is off, or the request was not confirmed."})
                return
            with lock:
                n = engine.self_model.clear()
            self.send_json(200, {"ok": True, "deleted": n})
        elif path.startswith("/api/memories/") or path == "/api/history/clear":
            self.memory_action(path, data, engine, lock)
        elif path.startswith("/api/knowledge/"):
            self.knowledge_action(path, data, engine, lock)
        else:
            self.send_json(404, {"error": "Not found."})

    def memory_action(self, path, data, engine, lock):
        """Your controls over long-term memory."""
        if engine.memory is None:
            self.send_json(400, {"error": "Long-term memory is off (the server was started with --no-memory)."})
            return
        memory_id = data.get("id")
        with lock:
            if path == "/api/memories/delete" and isinstance(memory_id, int):
                ok = engine.memory.delete(memory_id)
            elif path == "/api/memories/important" and isinstance(memory_id, int):
                ok = engine.memory.set_important(memory_id, bool(data.get("important")))
            elif path == "/api/memories/clear" and data.get("confirm") is True:
                ok = engine.memory.clear() >= 0
            elif path == "/api/history/clear" and data.get("confirm") is True:
                ok = engine.memory.clear_history() >= 0
            else:
                self.send_json(400, {"error": "The request could not be understood."})
                return
            if ok and engine.self_model is not None:
                action = {"/api/memories/delete": "deleted a memory", "/api/memories/clear": "deleted every memory",
                          "/api/history/clear": "deleted the conversation history",
                          "/api/memories/important": "marked a memory " + ("important" if data.get("important")
                                                                           else "not important")}[path]
                engine.self_model.control(action, {"memories": [memory_id]} if isinstance(memory_id, int) else {})
        if ok:
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "No such memory."})


    def knowledge_action(self, path, data, engine, lock):
        """Your controls over what was learned."""
        if engine.learning is None:
            self.send_json(400, {"error": "Learning is off (the server was started with --no-learning)."})
            return
        fact_id = data.get("id")
        with lock:
            if path == "/api/knowledge/remove" and isinstance(fact_id, int):
                ok = engine.learning.remove(fact_id)
            elif path == "/api/knowledge/clear" and data.get("confirm") is True:
                ok = engine.learning.clear() >= 0
            else:
                self.send_json(400, {"error": "The request could not be understood."})
                return
            if ok and engine.self_model is not None:
                engine.self_model.control("removed a learned fact" if path == "/api/knowledge/remove"
                                          else "removed everything learned",
                                          {"facts": [fact_id]} if isinstance(fact_id, int) else {})
        if ok:
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "No such fact."})

    def self_view(self, path, engine, lock):
        """The self-model's views. Reading never changes anything, and nothing here reaches the brain."""
        if engine.self_model is None:
            self.send_json(200, {"enabled": False})
            return
        if path == "/api/self/ask":
            query = dict(part.split("=", 1) for part in self.path.partition("?")[2].split("&") if "=" in part)
            question = unquote_plus(query.get("q", "")).strip()[:500]
            if not question:
                self.send_json(400, {"error": "There is no question."})
                return
            with lock:
                report = engine.self_model.ask(question)
            self.send_json(200, report or {"understood": False, "outside": False, "question": question,
                                           "note": "Not a question about recorded state (the forms the self-model "
                                                   "knows are listed in self_model/ask.py)."})
            return
        with lock:
            if path == "/api/self":
                self.send_json(200, engine.self_model.view())
            else:
                self.send_json(200, {"enabled": True, "events": engine.self_model.events()})


def create_server(model_path=None, port=DEFAULT_PORT, host=HOST, memory_path=DEFAULT_DB, use_memory=True,
                  knowledge_path=KNOWLEDGE_DB, use_learning=True, use_reasoning=True, events_path=EVENTS_DB,
                  use_self=True, state_path=STATE_DB, use_state=True, state_config=None):
    """Load the brain (default: the newest one) and build (but don't start) the server."""
    brain_file = Path(model_path) if model_path else newest_model()
    model, tokenizer, record = load_brain(brain_file)
    server = ThreadingHTTPServer((host, port), ChatHandler)
    memory = MemorySystem(memory_path, brain_name=record.get("stage")) if use_memory else None
    learning = LearningSystem(knowledge_path) if use_learning else None
    world = WorldSystem(learning.store, memory.store if memory else None) if learning and use_reasoning else None
    self_model = SelfSystem(events_path, persist=use_memory) if use_self else None     # memory off: RAM only
    internal = InternalState(state_path, persist=use_memory, config=state_config) if use_self and use_state else None
    server.engine = ChatEngine(model, tokenizer, record, memory=memory, learning=learning, world=world,
                               self_model=self_model, brain_file=brain_file, internal_state=internal)
    server.lock = threading.Lock()
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start the local chat page for our AI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--stage", choices=["1", "1.5", "1.5b", "1.5b2", "1.5b3", "1.5b4", "1.5b4b", "1.5b4c", "1.5b4d"],
                        help="which brain (default: the newest trained one)")
    parser.add_argument("--model", default=None, help="a specific trained model file")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--no-memory", action="store_true", help="chat without long-term memory")
    parser.add_argument("--memory-db", default=str(DEFAULT_DB), help="where long-term memory is kept")
    parser.add_argument("--no-learning", action="store_true", help="chat without learning from experience")
    parser.add_argument("--knowledge-db", default=str(KNOWLEDGE_DB), help="where learned knowledge is kept")
    parser.add_argument("--no-reasoning", action="store_true", help="without the world model and reasoning (Stage 2c)")
    parser.add_argument("--no-self", action="store_true", help="without the self-model and its event log")
    parser.add_argument("--events-db", default=str(EVENTS_DB), help="where the event log is kept")
    parser.add_argument("--no-state", action="store_true", help="without the internal state")
    parser.add_argument("--state-db", default=str(STATE_DB), help="where the internal state is kept")
    parser.add_argument("--state-memory-effects", action="store_true",
                        help="experimental: let the internal state raise memory importance and order recalled memories "
                             "(off by default: it lowered the memory benchmark, D-074)")
    args = parser.parse_args(argv)

    model_path = args.model or {"1": STAGE1_MODEL, "1.5": STAGE15_MODEL, "1.5b": STAGE15B_MODEL,
                                "1.5b2": STAGE15B2_MODEL, "1.5b3": STAGE15B3_MODEL, "1.5b4": STAGE15B4_MODEL,
                                "1.5b4b": STAGE15B4B_MODEL, "1.5b4c": STAGE15B4C_MODEL, "1.5b4d": STAGE15B4D_MODEL,
                                None: None}[args.stage]
    print("Starting our AI")
    try:
        server = create_server(model_path, args.port, memory_path=args.memory_db, use_memory=not args.no_memory,
                               knowledge_path=args.knowledge_db, use_learning=not args.no_learning,
                               use_reasoning=not args.no_reasoning, events_path=args.events_db,
                               use_self=not args.no_self, state_path=args.state_db, use_state=not args.no_state,
                               state_config={"memory": {"importance": True, "recall_order": True}}
                               if args.state_memory_effects else None)
    except ModelNotFoundError as e:
        print(f"\n{e}")
        return 1
    except OSError as e:
        print(f"\nCould not use port {args.port} ({e.strerror}). Something else is using it.\n"
              f"Try another port:\n    python -m chat.server --port {args.port + 1}")
        return 1

    engine = server.engine
    port = server.server_address[1]
    info = engine.info()
    markers = " + " + " ".join(engine.tokenizer.specials) if engine.tokenizer.specials else ""
    print(f"  brain:     Stage {engine.stage}, {info['model']}, {info['parameters']:,} parameters")
    print(f"  tokenizer: {len(engine.tokenizer.chars)} characters{markers}")
    print(f"  what it can do: {engine.capability()}")
    if engine.memory is not None:
        stored, history = len(engine.memory.memories()), engine.memory.history_counts()
        print(f"  long-term memory: {stored} memories, {history['turns']} saved messages "
              f"({engine.memory.path})")
    else:
        print("  long-term memory: OFF (nothing will be remembered after you stop)")
    if engine.learning is not None:
        counts = engine.learning.store.counts()
        known = sum(counts[c] for c in ("about_you", "taught", "general", "skill"))
        print(f"  learning: {known} learned facts, {counts['entities']} things known ({engine.learning.path})")
    else:
        print("  learning: OFF (nothing will be learned)")
    if engine.world is not None:
        counts = engine.world.view()["counts"]
        print(f"  world model + reasoning: ON ({counts['entities']} entities, {counts['relations']} relations learned)")
    else:
        print("  world model + reasoning: OFF")
    if engine.self_model is not None:
        log = engine.self_model.log
        where = engine.self_model.path if log.persistent else "RAM only, because memory is off"
        shown = ("only its answer to a question about recorded state is shown to the brain, as one line"
                 if engine.self_format == "self line" else "never shown to the brain")          # D-080
        print(f"  self-model + event log: ON ({log.count()} events, {len(log.runs())} runs; {where}; {shown})")
    else:
        print("  self-model + event log: OFF")
    if engine.internal_state is not None:
        state = engine.internal_state
        values = state.values()[0]
        print(f"  internal state: ON ({state.path if state.persistent else 'RAM only, because memory is off'}; "
              f"valence {values['valence']:+.2f}, activation {values['activation']:.2f}; never shown to the brain)")
    else:
        print("  internal state: OFF")
    print(f"\nAI running at http://localhost:{port}")
    print("Open that address in your browser. Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped." + (" Long-term memory is saved and will be there next time."
                              if engine.memory is not None else " (Memory was off: nothing was saved.)"))
    finally:
        server.server_close()
        if engine.internal_state is not None:
            engine.internal_state.close()
        if engine.self_model is not None:
            engine.self_model.stop()
            engine.self_model.close()
        if engine.memory is not None:
            engine.memory.close()
        if engine.learning is not None:
            engine.learning.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

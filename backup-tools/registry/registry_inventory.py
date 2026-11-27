"""Read-only local Study inventory for rollout planning."""
import argparse
import json
import sqlite3
from pathlib import Path

from global_registry import IDENTITY, Registry, now


def inspect(db):
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=10) as conn:
        studies = {}
        for sid, name in conn.execute("SELECT study_id, study_name FROM studies"):
            rows = {}
            for tid, number, state in conn.execute(
                "SELECT trial_id, number, state FROM trials WHERE study_id=? ORDER BY number", (sid,)
            ):
                params = dict(conn.execute(
                    "SELECT param_name, param_value FROM trial_params WHERE trial_id=?", (tid,)
                ))
                attrs = {
                    key: json.loads(value) for key, value in conn.execute(
                        "SELECT key, value_json FROM trial_system_attributes WHERE trial_id=? AND key='fixed_params'", (tid,)
                    )
                }
                rows[number] = dict(state=state, parameter_count=len(params), fixed_params=attrs.get("fixed_params"))
            studies[name] = rows
        return studies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("server_id")
    p.add_argument("root", type=Path)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--remote")
    args = p.parse_args()
    if not IDENTITY.fullmatch(args.server_id):
        p.error("invalid server ID")
    rows = {}
    for db in (*args.root.glob("*/optuna.db"), *args.root.glob("*/study.sqlite3")):
        for name, trials in inspect(db).items():
            status_file = db.parent / "search_status.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            active = status.get("active")
            rows[name] = {"trials": trials, "active": active}
    record = {"server_id": args.server_id, "observed_at": now(), "studies": rows,
              "inventory_only": True}
    print(json.dumps(record, indent=2))
    if args.publish:
        registry = Registry(args.remote) if args.remote else Registry()
        registry.update(Path("inventory") / f"{args.server_id}.json", lambda _: record)


if __name__ == "__main__":
    main()

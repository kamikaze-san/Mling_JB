"""
Group STRATA attack-log entries into conversations, correctly.

Why this exists (found 11 Nov): the lifelong and test logs store `iteration = 0` for EVERY pass, and
`request_id` restarts at 0 in every run. Grouping by (stage, request_id, iteration), as several scripts
did, silently merged the 8 conversations a goal has on its 8 surfaces (and goals of different runs) into
one, giving wrong contexts, wrong candidates and wrong held-out attacks. Each goal visits each surface once
per run, so the surface and the goal text identify a conversation. Entries repeated across logs (the
lifelong log carries the warm-up entries) are dropped.
"""
import hashlib


def conversation_key(e):
    return (e.get("stage"), e.get("request_id"), e.get("iteration"), e.get("profile"), e.get("request"))


def conversation_id(key):
    h = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:8]
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|{h}"


def group_conversations(entries):
    """Return ({key: [entries sorted by turn]}, n_duplicate_entries_dropped)."""
    groups, seen, dup = {}, set(), 0
    for e in entries:
        if not isinstance(e, dict) or "turn" not in e or "response" not in e:
            continue
        k = conversation_key(e)
        body = (e.get("clean_prompt") or e.get("prompt") or "") + "\x00" + (e.get("response") or "")
        sig = (k, e["turn"], bool(e.get("null_probe")), hashlib.sha1(body.encode("utf-8")).hexdigest())
        if sig in seen:
            dup += 1
            continue
        seen.add(sig)
        groups.setdefault(k, []).append(e)
    for turns in groups.values():
        turns.sort(key=lambda e: e["turn"])
    return groups, dup

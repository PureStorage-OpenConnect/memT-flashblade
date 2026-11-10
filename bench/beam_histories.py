"""BEAM-1M history construction for the E1 efficiency sweep (MemT paper Sec 6.5 / 6.6).

Records are rendered in the paper's BEAM format (Appendix D.3.1):
    [{session_id} | {timestamp}]
    User: {content}            or      Assistant: {content}
Sessions are numbered 1..S in chat order; assistant messages carry no time anchor in the
dataset ('None'), so they inherit the anchor of the latest user message of their session.
Records are joined with a blank line ("\n\n").

All lengths are token counts of the raw rendered history (no chat template). Token ids are
built by concatenating per-record tokenizations (+ separator ids), so the exact sequence fed
to both systems is known and identical; the text form is only for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BEAM_PARQUET = ("/mnt/data/robert-scripts/hf-cache/hub/datasets--Mohammadta--BEAM/snapshots/"
                "3205395e897e7318c7b094ef4e6047b9b82dbb03/data/1M-00000-of-00001.parquet")
SEP = "\n\n"


def load_beam(path=BEAM_PARQUET):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["conversation_id", "chat"])
    ids = t.column("conversation_id").to_pylist()
    chats = t.column("chat").to_pylist()
    return list(zip(ids, chats))


def render_records(chat):
    """chat = list of sessions, each a list of {role, content, time_anchor, ...} -> list[str]."""
    out = []
    for s_idx, session in enumerate(chat, start=1):
        anchor = None
        for m in session:
            ta = m.get("time_anchor")
            if ta and ta != "None":
                anchor = ta
            role = {"user": "User", "assistant": "Assistant"}.get(m["role"], m["role"].capitalize())
            out.append(f"[{s_idx} | {anchor or 'None'}]\n{role}: {m['content']}")
    return out


@dataclass
class TokenizedConversation:
    conv_id: str
    records: list            # list[list[int]], one per rendered record
    sep: list                # token ids of SEP
    n_records_total: int = 0
    tokens_available: int = 0


def tokenize_conversation(tok, conv_id, chat, max_tokens):
    """Tokenize records in order until at least max_tokens (+ separators) are covered."""
    recs = render_records(chat)
    sep = tok.encode(SEP, add_special_tokens=False)
    out, total = [], 0
    for r in recs:
        ids = tok.encode(r, add_special_tokens=False)
        out.append(ids)
        total += len(ids) + len(sep)
        if total >= max_tokens:
            break
    return TokenizedConversation(conv_id, out, sep, len(recs), total)


@dataclass
class History:
    conv_id: str
    target: int
    ids: list                         # flat raw history token ids
    pieces: list = field(default_factory=list)  # list[list[int]] record (possibly truncated) ids, no seps
    n_records: int = 0
    truncated_last: bool = False

    @property
    def n_tokens(self):
        return len(self.ids)


def build_history(tc: TokenizedConversation, target: int, tol: float = 0.01) -> History:
    """Whole records joined by SEP until ~target tokens. If the next whole record would overshoot
    target*(1+tol) and we are still below target*(1-tol), the next record is token-truncated so the
    total lands exactly on target."""
    lo, hi = int(target * (1 - tol)), int(target * (1 + tol))
    ids, pieces, n = [], [], 0
    trunc = False
    for rec in tc.records:
        add = (tc.sep if ids else []) + rec
        if len(ids) + len(add) <= hi:
            ids += add
            pieces.append(list(rec))
            n += 1
            if len(ids) >= lo:
                break
            continue
        if len(ids) >= lo:
            break
        room = target - len(ids) - (len(tc.sep) if ids else 0)
        if room > 0:
            ids += (tc.sep if ids else []) + rec[:room]
            pieces.append(list(rec[:room]))
            n += 1
            trunc = True
        break
    if len(ids) < lo:
        raise ValueError(f"{tc.conv_id}: only {len(ids)} tokens available for target {target}")
    return History(tc.conv_id, target, ids, pieces, n, trunc)


def chat_frame(tok):
    """Token ids around a single user message: prefix, end-of-user (no gen prompt), and the
    generation suffix with thinking disabled. Derived from the model's own chat template."""
    mark = "XQXSENTINELXQX"
    plain = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False)
    gen = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    pre, post_plain = plain.split(mark)
    pre2, post_gen = gen.split(mark)
    assert pre == pre2, (pre, pre2)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    return {"prefix": enc(pre), "end_user": enc(post_plain), "gen_suffix": enc(post_gen),
            "prefix_text": pre, "end_user_text": post_plain, "gen_suffix_text": post_gen}


def pack_slices(hist: History, frame, sep, cap: int = 2048):
    """Pack whole records into chat-formatted user-message slices of <= cap tokens.
    A record longer than one slice's content budget is split at token level across slices.
    Returns list of token-id lists (each = prefix + content + end_user)."""
    overhead = len(frame["prefix"]) + len(frame["end_user"])
    budget = cap - overhead
    slices, cur = [], []

    def flush():
        if cur:
            slices.append(frame["prefix"] + cur + frame["end_user"])

    for rec in hist.pieces:
        need = (len(sep) if cur else 0) + len(rec)
        if len(cur) + need <= budget:
            cur += (sep if cur else []) + rec
            continue
        if len(rec) <= budget:          # fits in a fresh slice
            flush()
            cur = list(rec)
            continue
        # oversized record: fill current slice, then spill in budget-sized pieces
        rest = list(rec)
        if cur:
            room = budget - len(cur) - len(sep)
            if room > 0:
                cur += sep + rest[:room]
                rest = rest[room:]
            flush()
            cur = []
        while len(rest) > budget:
            slices.append(frame["prefix"] + rest[:budget] + frame["end_user"])
            rest = rest[budget:]
        cur = rest
    flush()
    assert all(len(s) <= cap for s in slices)
    return slices


QUERY_INSTRUCTION = ("Based on everything you know about this user, write a detailed summary of their "
                     "situation, goals and preferences.")
_FILLER = ("Please organize the summary into clear sections, cover the most important facts first, mention "
           "any changes over time, note open questions or unresolved issues, keep a neutral and factual "
           "tone, avoid speculation beyond what is known, and use plain language that a new assistant "
           "could rely on when continuing to help this user. ")


def query_content_ids(tok, target=512):
    """Deterministic ~target-token query content: instruction + neutral filler, cut to exactly target."""
    text = QUERY_INSTRUCTION + "\n\n"
    while len(tok.encode(text, add_special_tokens=False)) < target + 32:
        text += _FILLER
    ids = tok.encode(text, add_special_tokens=False)[:target]
    return ids


def pack_slices_dense(hist: History, frame, cap: int = 2048):
    """Alternative: split the flat raw history (records + separators) into contiguous content pieces
    of exactly (cap - template overhead) tokens, ignoring message boundaries."""
    budget = cap - len(frame["prefix"]) - len(frame["end_user"])
    return [frame["prefix"] + hist.ids[i:i + budget] + frame["end_user"] for i in range(0, len(hist.ids), budget)]

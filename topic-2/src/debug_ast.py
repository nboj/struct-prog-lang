# debug_ast.py
from __future__ import annotations
from dataclasses import is_dataclass, fields
from typing import Any, Iterable, Optional
from src.source_map import SourceMap
import enum

def render_ast(root: Any, *, show_spans: bool = True,
               show_token_lexeme: bool = True,
               source_map: Optional["SourceMap"] = None) -> str:
    lines: list[str] = []
    _render(root, lines, prefix="", is_last=True,
            show_spans=show_spans,
            show_token_lexeme=show_token_lexeme,
            sm=source_map, seen=set())
    return "\n".join(lines)

def _render(obj: Any, out: list[str], *, prefix: str, is_last: bool,
            show_spans: bool, show_token_lexeme: bool,
            sm: Optional["SourceMap"], seen: set[int]) -> None:
    branch = "└─ " if is_last else "├─ "
    out.append(prefix + branch + _label(obj, show_spans, show_token_lexeme, sm))

    oid = id(obj)
    if oid in seen:
        out.append(prefix + ("   " if is_last else "│  ") + "↩︎ (seen)")
        return
    seen.add(oid)

    child_prefix = prefix + ("   " if is_last else "│  ")
    kids = list(_children(obj))
    for i, (name, child) in enumerate(kids):
        last = (i == len(kids) - 1)
        head = child_prefix + ("└─ " if last else "├─ ")
        if child.__class__.__name__ == "Span" and not show_spans:
            continue
        if _is_atomic(child):
            out.append(head + f"{name} = " +
                       _label(child, show_spans, show_token_lexeme, sm))
        else:
            out.append(head + f"{name}:")
            _render(child, out, prefix=child_prefix, is_last=last,
                    show_spans=show_spans,
                    show_token_lexeme=show_token_lexeme,
                    sm=sm, seen=seen)

def _label(obj: Any, show_spans: bool, show_token_lexeme: bool,
           sm: Optional["SourceMap"]) -> str:
    if obj is None:
        return "None"
    if isinstance(obj, (str, int, float, bool)):
        return repr(obj)
    if isinstance(obj, enum.Enum):
        return f"{obj.__class__.__name__}.{obj.name}"

    cls = obj.__class__.__name__

    # Token rendering
    if cls == "Token":
        kind = getattr(obj, "kind", None)
        raw  = getattr(obj, "raw", None)
        parts = [getattr(kind, "name", str(kind))]
        if show_token_lexeme and raw is not None:
            parts.append(repr(raw))
        if show_spans and hasattr(obj, "span"):
            parts.append(_span_str(getattr(obj, "span"), sm))
        return "Token(" + ", ".join(parts) + ")"

    # Span rendering
    if cls == "Span":
        return "Span(" + _span_str(obj, sm) + ")"

    # Sequences
    if isinstance(obj, (list, tuple)):
        return f"{cls}[{len(obj)}]"

    # Dataclass AST nodes
    if is_dataclass(obj):
        name = obj.__class__.__name__
        if show_spans and hasattr(obj, "span"):
            return f"{name} [{_span_str(getattr(obj, 'span'), sm)}]"
        return name

    return repr(obj)

def _children(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, (list, tuple)):
        for i, el in enumerate(obj):
            yield (f"[{i}]", el)
        return
    if is_dataclass(obj):
        for f in fields(obj):
            yield (f.name, getattr(obj, f.name))
        return
    # Tokens/Spans: no children

def _is_atomic(x: Any) -> bool:
    if x is None or isinstance(x, (str, int, float, bool, enum.Enum)):
        return True
    if x.__class__.__name__ in ("Token", "Span"):
        return True
    return not (is_dataclass(x) or isinstance(x, (list, tuple)))

def _span_str(span: Any, sm: Optional["SourceMap"]) -> str:
    try:
        if sm is not None:
            (sl, sc), (el, ec) = sm.span_to_lc(span)
            return f"{sl}:{sc}-{el}:{ec}"
        return f"{span.start}..{span.end}"
    except Exception:
        return "?"

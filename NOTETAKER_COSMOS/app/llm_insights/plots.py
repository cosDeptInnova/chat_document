from __future__ import annotations

import io
from threading import Lock
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless server backend

import matplotlib.pyplot as plt


_PLOT_LOCK = Lock()


def _sort_items(d: Dict[str, float]) -> List[Tuple[str, float]]:
    items = [(k, float(v)) for k, v in (d or {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def render_insights_png(insights: Dict[str, Any], title: str = "LLM Meeting Insights") -> bytes:
    """
    Genera un PNG (bytes) con gráficos de:
      - participation_percent
      - talk_time_seconds
      - turns
      - collaboration / decisiveness / conflict_level
      - atmosphere.valence
      - topics (si existen)
      - caja de texto con summary/flags/decisions/action_items
    """
    with _PLOT_LOCK:
        participation = insights.get("participation_percent") or {}
        talk_time = insights.get("talk_time_seconds") or {}
        turns = insights.get("turns") or {}

        collab = (insights.get("collaboration") or {}).get("score_0_100", None)
        decis = (insights.get("decisiveness") or {}).get("score_0_100", None)
        conflict = insights.get("conflict_level_0_100", None)
        valence = (insights.get("atmosphere") or {}).get("valence", None)

        topics = insights.get("topics") or []
        decisions = insights.get("decisions") or []
        action_items = insights.get("action_items") or []
        qflags = insights.get("quality_flags") or []

        fig = plt.figure(figsize=(14, 10), dpi=140)
        fig.suptitle(title, fontsize=16)

        ax1 = fig.add_axes([0.06, 0.56, 0.42, 0.32])  # participation
        ax2 = fig.add_axes([0.52, 0.56, 0.42, 0.32])  # talk time
        ax3 = fig.add_axes([0.06, 0.22, 0.42, 0.28])  # turns
        ax4 = fig.add_axes([0.52, 0.22, 0.42, 0.28])  # scores/topics
        ax5 = fig.add_axes([0.06, 0.05, 0.88, 0.12])  # text
        ax5.axis("off")

        # participation
        part_items = _sort_items(participation)
        ax1.set_title("Participation (%)")
        if part_items:
            labels = [k for k, _ in part_items]
            values = [v for _, v in part_items]
            ax1.barh(labels, values)
            ax1.invert_yaxis()
            ax1.set_xlabel("%")
            ax1.set_xlim(0, max(100.0, max(values) * 1.1))
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        # talk time
        tt_items = _sort_items(talk_time)
        ax2.set_title("Talk time (seconds)")
        if tt_items:
            labels = [k for k, _ in tt_items]
            values = [v for _, v in tt_items]
            ax2.barh(labels, values)
            ax2.invert_yaxis()
            ax2.set_xlabel("seconds")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        # turns
        turns_items = _sort_items({k: float(v) for k, v in turns.items()})
        ax3.set_title("Turns (#)")
        if turns_items:
            labels = [k for k, _ in turns_items]
            values = [v for _, v in turns_items]
            ax3.barh(labels, values)
            ax3.invert_yaxis()
            ax3.set_xlabel("turns")
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

        # scores/topics
        ax4.set_title("LLM Scores & Topics")
        ax4.set_xlim(0, 100)
        ax4.set_ylim(-1, 6)
        ax4.set_yticks([])

        rows = []
        if collab is not None:
            rows.append(("Collaboration", float(collab)))
        if decis is not None:
            rows.append(("Decisiveness", float(decis)))
        if conflict is not None:
            rows.append(("Conflict", float(conflict)))

        y = 5
        for name, v in rows:
            ax4.barh([y], [v], height=0.6)
            ax4.text(1, y, name, va="center")
            ax4.text(min(99, v + 1), y, f"{v:.1f}", va="center")
            y -= 1

        # valence (-1..1) map to 0..100
        if valence is not None:
            ax4.axvline(50, linestyle="--", linewidth=1)
            v = max(-1.0, min(1.0, float(valence)))
            mapped = (v + 1.0) * 50.0
            ax4.scatter([mapped], [y], s=60)
            ax4.text(1, y, "Atmosphere valence (-1..1)", va="center")
            ax4.text(min(99, mapped + 1), y, f"{v:.2f}", va="center")
            y -= 1

        # topics (top 5)
        t_items = []
        for t in topics:
            try:
                name = str(t.get("name", "")).strip()
                w = float(t.get("weight", 0.0))
                if name:
                    t_items.append((name, w))
            except Exception:
                continue
        t_items.sort(key=lambda x: x[1], reverse=True)
        t_items = t_items[:5]

        if t_items:
            ax4.text(1, y, "Top topics", va="center")
            y -= 0.8
            for name, w in t_items:
                mapped_w = max(0.0, min(1.0, w)) * 100.0
                ax4.barh([y], [mapped_w], height=0.45)
                ax4.text(1, y, name[:40], va="center")
                ax4.text(min(99, mapped_w + 1), y, f"{w:.2f}", va="center")
                y -= 0.7

        # text box
        summary = str(insights.get("summary") or "").strip()
        if len(summary) > 600:
            summary = summary[:600] + "…"

        lines = [
            f"Decisions: {len(decisions)} | Action items: {len(action_items)} | Quality flags: {len(qflags)}"
        ]
        if qflags:
            lines.append("Flags: " + ", ".join([str(x) for x in qflags[:8]]))
        if summary:
            lines.append("Summary: " + summary)

        ax5.text(0.0, 0.8, "\n".join(lines), fontsize=10, va="top")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

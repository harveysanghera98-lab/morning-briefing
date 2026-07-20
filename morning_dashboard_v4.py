#!/usr/bin/env python3
"""
morning_dashboard_v4.py — Personal Morning Briefing Dashboard Generator

Improvements over v3:
  · 5th section: Deep Dive — rotating long-form wildcard analysis
  · Config-driven via config.yaml (falls back to built-in defaults)
  · CSS fixes: all flex gap values now have correct px units
  · Improved spacing and visual breathing room throughout
  · Wildcard history tracking prevents topic repetition (30-day window)
  · --force flag to regenerate an existing day's briefing
  · GitHub Actions ready — see .github/workflows/briefing.yml

Four API calls per run:
  1. Claude Sonnet + web search  — top stories, sectors, week ahead
  2. Claude Sonnet + web search  — deep dive (rotating long-form)
  3. Claude Sonnet + web search  — signal scan (voices RSS + X fallback)
  4. Claude Haiku (no search)    — pattern watch (~20x cheaper)

Setup:
    pip3 install anthropic pyyaml feedparser
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
    python3 morning_dashboard_v4.py
    python3 morning_dashboard_v4.py --config /path/to/config.yaml
    python3 morning_dashboard_v4.py --force   # regenerate today even if file exists
"""

import anthropic
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# ── Runtime constants ────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_RETRIES = 3

# ── Cost profile ──────────────────────────────────────────────────────────────
# BRIEFING_PROFILE=measure  → no search caps, deep dive runs daily. Reproduces the
#                             original behaviour so you can log a true baseline.
# BRIEFING_PROFILE=economy  → apply web-search caps + deep-dive schedule from config.
# Default is economy. Cost logging is always on in both modes.
BRIEFING_PROFILE = os.environ.get("BRIEFING_PROFILE", "economy").lower()

# Per-token prices in USD. Estimates for mid-2026 — verify at console.anthropic.com.
# Only relative attribution matters for tuning; absolute totals track console within
# rounding as long as these are current.
PRICING = {
    "claude-sonnet-4-6":         {"in": 3.00e-6, "out": 15.00e-6, "cache_read": 0.30e-6, "cache_write": 3.75e-6},
    "claude-haiku-4-5-20251001": {"in": 1.00e-6, "out":  5.00e-6, "cache_read": 0.10e-6, "cache_write": 1.25e-6},
}
WEB_SEARCH_COST = 10.0 / 1000  # USD per web-search request (server tool)

_USAGE_LOCK = threading.Lock()
_USAGE_TOTALS = {"cost": 0.0, "input": 0, "output": 0, "cache_read": 0, "searches": 0}


def _track_usage(label, model, response):
    """Record token usage and estimated cost from an API response. Thread-safe."""
    u = getattr(response, "usage", None)
    if u is None:
        return
    inp        = getattr(u, "input_tokens", 0) or 0
    out        = getattr(u, "output_tokens", 0) or 0
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_wr   = getattr(u, "cache_creation_input_tokens", 0) or 0
    searches   = 0
    stu = getattr(u, "server_tool_use", None)
    if stu is not None:
        searches = getattr(stu, "web_search_requests", 0) or 0
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    cost = (inp * p["in"] + out * p["out"]
            + cache_read * p["cache_read"] + cache_wr * p["cache_write"]
            + searches * WEB_SEARCH_COST)
    with _USAGE_LOCK:
        _USAGE_TOTALS["cost"]       += cost
        _USAGE_TOTALS["input"]      += inp + cache_read + cache_wr
        _USAGE_TOTALS["output"]     += out
        _USAGE_TOTALS["cache_read"] += cache_read
        _USAGE_TOTALS["searches"]   += searches
    print(f"    [cost] {label}: {inp + cache_read + cache_wr:,} in / {out:,} out / "
          f"{searches} search(es) ~ ${cost:.3f}")


def print_usage_summary():
    """Print the run's token + cost totals."""
    t = _USAGE_TOTALS
    total = t["input"] + t["output"]
    print("\n-- Cost summary ---------------------------------------")
    print(f"  Backend:       {BRIEFING_BACKEND}")
    print(f"  Profile:       {BRIEFING_PROFILE}")
    print(f"  Total tokens:  {total:,}  ({t['input']:,} in / {t['output']:,} out)")
    print(f"  Cache reads:   {t['cache_read']:,} tokens (billed ~0.1x)")
    print(f"  Web searches:  {t['searches']}")
    print(f"  Est. cost:     ${t['cost']:.2f}")
    print("  (prices are estimates — confirm against console.anthropic.com)")


def _search_caps(cfg):
    """Merge configured web-search caps over the built-in defaults."""
    caps = {}
    if isinstance(cfg, dict):
        caps = cfg.get("cost", {}).get("search_caps", {}) or {}
    merged = dict(DEFAULT_CONFIG["cost"]["search_caps"])
    merged.update(caps)
    return merged


def web_search_tool(kind, cfg):
    """Web-search tool spec. In economy mode, cap searches via max_uses."""
    tool = {"type": "web_search_20250305", "name": "web_search"}
    if BRIEFING_PROFILE == "economy":
        cap = _search_caps(cfg).get(kind)
        if cap:
            tool["max_uses"] = cap
    return tool


def _deep_dive_due(cfg):
    """Whether to run the search-heavy deep dive today."""
    if BRIEFING_PROFILE != "economy":
        return True
    freq = "daily"
    if isinstance(cfg, dict):
        freq = cfg.get("cost", {}).get("deep_dive_frequency", "daily")
    if freq == "alternate":
        return datetime.date.today().toordinal() % 2 == 0
    return True


# ── Model backend: API key vs Claude Code (Pro subscription) ──────────────────
# BRIEFING_BACKEND=api  → call the Anthropic API with ANTHROPIC_API_KEY (per-token).
# BRIEFING_BACKEND=pro  → run each call through the Claude Code CLI authenticated
#                         with CLAUDE_CODE_OAUTH_TOKEN (draws on your subscription),
#                         and fall back to the API automatically on any failure.
BRIEFING_BACKEND = os.environ.get("BRIEFING_BACKEND", "api").lower()
CLAUDE_CODE_OAUTH_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
_PRO_WARNED = False


def _cc_model(model):
    """Map a full model id to a Claude Code alias."""
    m = (model or "").lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"


def _track_pro_usage(label, env_json):
    """Record usage/cost from a Claude Code JSON envelope. Cost is ~$0 on a
    subscription — that's the saving showing up in the summary."""
    cost = 0.0
    inp = out = 0
    if isinstance(env_json, dict):
        cost = env_json.get("total_cost_usd") or 0.0
        usage = env_json.get("usage") or {}
        inp = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
        out = usage.get("output_tokens") or 0
    with _USAGE_LOCK:
        _USAGE_TOTALS["cost"]   += cost
        _USAGE_TOTALS["input"]  += inp
        _USAGE_TOTALS["output"] += out
    print(f"    [pro] {label}: {inp:,} in / {out:,} out ~ ${cost:.3f} (subscription)")


def _run_via_claude_code(label, model, prompt, system, use_search, max_turns):
    """Run one generation through the Claude Code CLI on the Pro subscription.
    Raises on any failure so the caller can fall back to the API."""
    cmd = ["claude", "-p", prompt,
           "--output-format", "json",
           "--model", _cc_model(model),
           "--max-turns", str(max_turns)]
    if system:
        cmd += ["--append-system-prompt", system]
    if use_search:
        cmd += ["--allowedTools", "WebSearch"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=900, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {(proc.stderr or '')[-300:]}")
    try:
        env_json = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise ValueError(f"claude output not JSON: {(proc.stdout or '')[:200]}")
    if isinstance(env_json, dict) and env_json.get("is_error"):
        raise RuntimeError(f"claude reported error: {str(env_json.get('result', ''))[:200]}")
    text = env_json.get("result", "") if isinstance(env_json, dict) else ""
    if not text or not text.strip():
        raise ValueError("empty result from claude code")
    _track_pro_usage(label, env_json)
    return text


def _run_via_api(label, model, prompt, system, max_tokens, use_search, cfg, search_kind):
    """Run one generation through the Anthropic API with the API key."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if use_search:
        kwargs["tools"] = [web_search_tool(search_kind, cfg)]
    response = client.messages.create(**kwargs)
    _track_usage(label, model, response)
    parts = [b.text for b in response.content if hasattr(b, "text") and b.text.strip()]
    return "\n".join(parts)


def run_model(*, label, model, prompt, max_tokens, use_search, cfg,
              system=None, search_kind=None, max_turns=6):
    """Return assistant text. Prefer the Pro path when configured; always fall
    back to the API so a run still ships if the subscription path fails."""
    global _PRO_WARNED
    if BRIEFING_BACKEND == "pro":
        if not CLAUDE_CODE_OAUTH_TOKEN:
            if not _PRO_WARNED:
                print("    [pro] BRIEFING_BACKEND=pro but CLAUDE_CODE_OAUTH_TOKEN unset — using API")
                _PRO_WARNED = True
        else:
            try:
                return _run_via_claude_code(label, model, prompt, system, use_search, max_turns)
            except Exception as e:
                print(f"    [pro] {label} failed ({e}); falling back to API")
    return _run_via_api(label, model, prompt, system, max_tokens,
                        use_search, cfg, search_kind)

# ── Default config (Harvey's setup — used when no config.yaml is found) ──────
DEFAULT_CONFIG = {
    "topics": [
        {
            "name": "Frontier Tech / AI",
            "description": (
                "AI, quantum computing, semiconductors/GPU supply chain, compute "
                "infrastructure, photonics, bio/biotech, space/launch, robotics"
            ),
        },
        {
            "name": "Applied Tech & Industry",
            "description": (
                "Big Tech strategy, startups & VC, enterprise AI adoption, "
                "major M&A/funding/IPOs/earnings, platform shifts, talent signals"
            ),
        },
        {
            "name": "US Policy & Power",
            "description": (
                "Tech regulation, fiscal/monetary, defense & industrial policy, "
                "White House & Congress, budget, judicial decisions, elections, science policy"
            ),
        },
        {
            "name": "AI Regulation & Governance",
            "description": (
                "US executive orders, EU AI Act, global frameworks, "
                "industry self-regulation, safety institutes"
            ),
        },
        {
            "name": "Geopolitics",
            "description": "US-China, Middle East, India, UK/EU, Latin America, Africa",
        },
        {
            "name": "China",
            "description": "Tech decoupling, economic health, Taiwan, industrial overcapacity",
        },
        {
            "name": "Markets",
            "description": (
                "Equities, rates/Fed, VC, commodities, central banks, "
                "inflation, bonds, currencies"
            ),
        },
        {
            "name": "Digital Finance",
            "description": "Crypto regulation, CBDCs, stablecoins, DeFi",
        },
        {
            "name": "Energy",
            "description": "Oil & gas, nuclear/renewables, AI infrastructure, grid",
        },
        {
            "name": "Defense & Security",
            "description": "Conflicts, military tech, cyber, intelligence",
        },
        {
            "name": "Labor & Workforce",
            "description": "AI displacement, immigration/talent, unions, remote work",
        },
    ],
    "sources": {
        "preferred": [
            "Financial Times",
            "Wall Street Journal",
            "Semafor",
            "Reuters",
            "Al Jazeera",
            "AP",
            "The Economist",
            "The Atlantic",
        ],
        "supplementary": ["Ars Technica", "TechCrunch", "Defense One"],
    },
    "sections": {
        "top_stories": True,
        "sector_scan": True,
        "week_ahead": True,
        "pattern_watch": True,
        "deep_dive": True,
        "signal_scan": True,
    },
    "signal_scan_frequency": "weekly",  # "daily" or "weekly"
    "voices": [
        {"name": "Leopold Aschenbrenner", "rss": "https://situational.substack.com/feed",          "x_handle": "leopoldasch"},
        {"name": "Dan Wang",              "rss": "https://danwwang.substack.com/feed",              "x_handle": "dkwang"},
        {"name": "Nathan Benaich",        "rss": "https://nathanbenaich.substack.com/feed",         "x_handle": "nathanbenaich"},
        {"name": "Stanley Druckenmiller", "rss": None,                                              "x_handle": None,
         "search_query": "Stanley Druckenmiller interview remarks site:cnbc.com OR site:bloomberg.com OR site:ft.com"},
        {"name": "Byrne Hobart",          "rss": "https://thediff.co/feed",                        "x_handle": "byrnehobart"},
        {"name": "Michael Burry",         "rss": None,                                              "x_handle": "michaeljburry"},
        {"name": "Ben Thompson",          "rss": "https://stratechery.com/feed/",                   "x_handle": "stratechery"},
        {"name": "Paul Graham",           "rss": "https://www.aaronsw.com/2002/feeds/pgessays.rss", "x_handle": "paulg"},
        {"name": "Benedict Evans",        "rss": "https://www.ben-evans.com/benedictevans/rss.xml", "x_handle": "benedictevans"},
        {"name": "Packy McCormick",       "rss": "https://www.notboring.co/feed",                   "x_handle": "packym"},
        {"name": "CJ Gustafson",          "rss": "https://www.mostlymetrics.com/feed",              "x_handle": "cjgus"},
        {"name": "Lenny Rachitsky",       "rss": "https://www.lennysnewsletter.com/feed",           "x_handle": "lennysan"},
        {"name": "Andrej Karpathy",       "rss": "https://karpathy.github.io/feed.xml",             "x_handle": "karpathy"},
        {"name": "Ilya Sutskever",        "rss": None,                                              "x_handle": "ilyasut"},
        {"name": "Noam Brown",            "rss": None,                                              "x_handle": "noambrown"},
        {"name": "François Chollet",      "rss": None,                                              "x_handle": "fchollet"},
    ],
    "output": {
        "dir": "~/briefings",
    },
    "cost": {
        # Max web searches per call in economy mode. Lower = cheaper, less coverage.
        "search_caps": {"main": 3, "deep_dive": 3, "signal": 10},
        # "daily" or "alternate" (deep dive runs every other calendar day).
        "deep_dive_frequency": "alternate",
    },
}


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path=None):
    """Load config.yaml if present, else return built-in defaults."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )

    if os.path.exists(config_path):
        if not YAML_AVAILABLE:
            print(
                "Warning: config.yaml found but pyyaml not installed. "
                "Run: pip install pyyaml\nUsing built-in defaults."
            )
            return DEFAULT_CONFIG
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        # Backfill any keys not present in the user's config
        for key, val in DEFAULT_CONFIG.items():
            if key not in cfg:
                cfg[key] = val
        return cfg

    return DEFAULT_CONFIG


# ── Config-aware prompt builders ──────────────────────────────────────────────

def build_system_prompt(cfg):
    topics = cfg["topics"]
    preferred = cfg["sources"]["preferred"]
    supplementary = cfg["sources"].get("supplementary", [])

    topic_lines = "\n".join(
        f"{i + 1}. {t['name']} — {t['description']}"
        for i, t in enumerate(topics)
    )
    preferred_str = ", ".join(preferred)
    supp_str = ", ".join(supplementary) if supplementary else "specialist trade outlets"

    return f"""You are my daily intelligence briefer. Each morning, search the web thoroughly
and deliver a concise, substantive briefing on what happened in the last 24 hours
and what's ahead this week.

MY TOPIC PRIORITIES (in order):

{topic_lines}

SEARCH APPROACH: Use 4-6 targeted searches across major outlets. Combine related
topics into single queries where possible. Prioritize original reporting.

PREFERRED SOURCES (weight these heavily):
{preferred_str}. Use these as primary sources wherever possible.
Supplement with specialist/trade outlets (e.g. {supp_str})
only when the preferred sources don't cover a topic.
"""


def build_format_instructions(cfg):
    topics = cfg["topics"]
    topic_names_quoted = ", ".join(f'"{t["name"]}"' for t in topics)
    n = len(topics)

    return f"""CRITICAL: Respond with ONLY valid JSON. No markdown, no explanation, no preamble,
no code fences. Just the raw JSON object.

{{
  "top_stories": [
    {{
      "priority": "urgent" | "high" | "tracking",
      "headline": "Short punchy headline",
      "one_liner": "One sentence on why this matters right now",
      "detail": "3-5 paragraphs. What happened, why it matters, where perspectives diverge (name specific outlets), what to watch next. Plain text, no markdown.",
      "sector": "Which of my {n} sectors this belongs to",
      "sources": [
        {{"name": "Publication Name", "url": "https://exact-article-url.com"}}
      ]
    }}
  ],
  "sectors": [
    {{
      "name": "Frontier Tech / AI",
      "has_news": true | false,
      "summary": "2-4 sentences if there is news. Empty string if quiet."
    }}
  ],
  "week_ahead": [
    {{
      "day": "Mon 6",
      "event": "Description of scheduled event",
      "tag": "critical" | "defense" | "space" | "data" | "earnings" | "policy" | "tech" | "other"
    }}
  ]
}}

REQUIREMENTS:
- top_stories: 3-5 items. At least 1 "urgent" if genuinely urgent news exists.
- sectors: All {n} topic areas. Exact names: {topic_names_quoted}.
  Set has_news to false and summary to "" if nothing meaningful happened.
- week_ahead: 6-12 scheduled events for the rest of this week.
- sources per story: 2-4 items. Use the actual URLs you retrieved during your web searches.
  Only include URLs you actually visited — do not fabricate them.

ONLY output the JSON. Nothing else.
"""


def build_deep_dive_prompt(cfg, history):
    topics_str = ", ".join(t["name"] for t in cfg["topics"])
    preferred_str = ", ".join(cfg["sources"]["preferred"])
    history_str = (
        "\n".join(f"- {h['date']}: {h['topic']}" for h in history[-14:])
        or "None yet."
    )

    return f"""You are writing a long-form intelligence brief for a sophisticated reader.

RECENT DEEP DIVE TOPICS — do not repeat any of these:
{history_str}

Pick ONE specific, timely subject from today's news that:
- Is genuinely interesting and topical RIGHT NOW — anchored in something that happened today or this week
- Intersects with these interest areas: {topics_str}
- Has real depth: competing interpretations, significant stakes, non-obvious angles
- Has NOT been covered recently (see list above)

Preferred sources: {preferred_str}

Search thoroughly, then write a long-form brief of 500-700 words. Cover:
1. The specific news hook — what happened today or this week
2. Deeper context and history that makes it matter
3. Competing perspectives — what reasonable people disagree about, and why
4. What to watch for next, and on what timeline

Return ONLY valid JSON — no markdown, no preamble, no code fences:
{{
  "topic": "Brief topic label, 5-10 words",
  "headline": "Sharp, specific headline",
  "standfirst": "One punchy sentence — the core argument or key fact",
  "body": "Full text, 500-700 words. Separate paragraphs with double newlines (\\n\\n). Plain text only — no markdown.",
  "why_today": "One sentence: why this subject is specifically relevant today",
  "sources": [
    {{"name": "Publication Name", "url": "https://exact-article-url.com"}}
  ]
}}
"""


# ── Pattern watch prompt ──────────────────────────────────────────────────────
PATTERN_PROMPT_BASE = """You are an intelligence analyst. Given the briefing data below, generate
pattern_watch analyses: speculative, cross-sector dot-connecting. Bold and opinionated.
Sharp wrong takes beat safe observations.

Return ONLY a valid JSON array — no markdown, no preamble, no code fences.

[
  {
    "title": "Short punchy title",
    "subtitle": "One-line thesis",
    "categories": ["Sector1", "Sector2"],
    "content": "2-3 paragraphs of speculative analysis connecting dots across sectors. Label as speculative. Plain text, no markdown."
  }
]

Requirements:
- 3-4 items
- Each must connect at least 2 sectors from the briefing
- Be bold and specific — name companies, people, policies
- Sharp wrong takes beat safe observations

Briefing data:
"""


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Morning Briefing — {{DATE}}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Source+Serif+4:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
    max-width: 720px; margin: 0 auto; padding: 0 24px 80px;
    color: #111827; background: #fff; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ─────────────────────────────────────────────── */
  .header { padding-top: 40px; margin-bottom: 32px; }
  .header-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    color: #9CA3AF; text-transform: uppercase; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }
  .header h1 {
    font-size: 30px; font-weight: 700; color: #111827;
    font-family: 'Source Serif 4', Georgia, serif;
    line-height: 1.2; margin-bottom: 14px;
  }
  .status-bar {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .status-urgent {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 600; color: #DC2626;
    background: #FEE2E2; padding: 4px 12px; border-radius: 4px;
  }
  .pulse-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #DC2626;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .status-count { font-size: 12px; color: #6B7280; }

  /* ── Tabs ────────────────────────────────────────────────── */
  .tabs {
    display: flex; gap: 2px; border-bottom: 1px solid #E5E7EB;
    margin-bottom: 28px; overflow-x: auto; -webkit-overflow-scrolling: touch;
  }
  .tab {
    padding: 10px 16px; font-size: 13px; font-weight: 500; color: #6B7280;
    background: none; border: none; border-bottom: 2px solid transparent;
    cursor: pointer; font-family: inherit; margin-bottom: -1px;
    white-space: nowrap; transition: color 0.15s ease;
  }
  .tab:hover { color: #374151; }
  .tab.active { font-weight: 700; color: #111827; border-bottom-color: #111827; }
  .arc-nav-link {
    margin-left: auto; padding: 9px 0 9px 12px;
    font-size: 12px; color: #6B7280; text-decoration: none;
    white-space: nowrap; display: flex; align-items: center; gap: 5px;
    flex-shrink: 0; transition: color 0.15s ease;
  }
  .arc-nav-link:hover { color: #374151; }
  .arc-nav-link svg { width: 13px; height: 13px; }
  .section { display: none; }
  .section.active { display: block; }
  .hint { font-size: 13px; color: #6B7280; margin-bottom: 20px; }

  /* ── Story cards ─────────────────────────────────────────── */
  .story {
    border: 1.5px solid #D1D5DB; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
    cursor: pointer; background: #FAFAFA;
    transition: border-color 0.2s ease, background 0.2s ease;
    overflow: hidden;
  }
  .story:hover { border-color: #9CA3AF; }
  .story.expanded-urgent  { background: #FEF2F2; border-color: #DC2626; }
  .story.expanded-high    { background: #FFFBEB; border-color: #D97706; }
  .story.expanded-tracking{ background: #F9FAFB; border-color: #6B7280; }
  .story-top { display: flex; align-items: flex-start; gap: 16px; }
  .story-dot {
    width: 8px; height: 8px; border-radius: 50%;
    margin-top: 13px; flex-shrink: 0;
  }
  .dot-urgent   { background: #DC2626; }
  .dot-high     { background: #D97706; }
  .dot-tracking { background: #6B7280; }
  .story-content { flex: 1; min-width: 0; }
  .story-meta {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px; flex-wrap: wrap;
  }
  .priority-tag {
    font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
    padding: 2px 8px; border-radius: 3px; text-transform: uppercase;
  }
  .tag-urgent   { color: #991B1B; background: #FEE2E2; border: 1px solid #FECACA; }
  .tag-high     { color: #92400E; background: #FEF3C7; border: 1px solid #FDE68A; }
  .tag-tracking { color: #374151; background: #F3F4F6; border: 1px solid #D1D5DB; }
  .sector-label {
    font-size: 9px; font-weight: 500; color: #9CA3AF;
    letter-spacing: 0.06em; text-transform: uppercase;
  }
  .story h3 {
    font-size: 17px; font-weight: 700; color: #111827;
    margin: 0 0 6px; line-height: 1.3;
    font-family: 'Source Serif 4', Georgia, serif;
  }
  .story .one-liner { font-size: 13.5px; color: #4B5563; line-height: 1.5; }
  .chevron {
    font-size: 18px; color: #9CA3AF; margin-top: 8px; flex-shrink: 0;
    transition: transform 0.25s ease;
  }
  .story.expanded .chevron { transform: rotate(180deg); }
  .story-detail {
    max-height: 0; overflow: hidden; transition: max-height 0.5s ease;
  }
  .story.expanded .story-detail { max-height: 3000px; }
  .story-detail-inner {
    margin-top: 20px; padding-top: 18px;
    border-top: 1px solid rgba(0,0,0,0.07);
  }
  .story-detail p {
    font-size: 14px; color: #1F2937; line-height: 1.8;
    margin-bottom: 16px; padding-left: 24px;
  }
  .story-detail p:last-child { margin-bottom: 4px; }

  /* ── Sector items ────────────────────────────────────────── */
  .sector-item {
    padding: 14px 18px; border-radius: 8px; cursor: pointer;
    background: #FAFAFA; border: 1px solid #E5E7EB;
    transition: border-color 0.2s ease, background 0.2s ease;
    margin-bottom: 8px;
  }
  .sector-item:hover { border-color: #9CA3AF; background: #F9FAFB; }
  .sector-item.expanded { background: #EFF6FF; border-color: #3B82F6; }
  .sector-top { display: flex; align-items: center; gap: 14px; }
  .sector-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .sector-dot.active { background: #10B981; }
  .sector-dot.quiet  { background: transparent; border: 1.5px solid #D1D5DB; }
  .sector-name { font-size: 13.5px; font-weight: 600; color: #1F2937; }
  .sector-summary {
    font-size: 13.5px; color: #374151; line-height: 1.75;
    margin: 12px 0 2px 23px; display: none;
  }
  .sector-item.expanded .sector-summary { display: block; }

  /* ── Week ahead ──────────────────────────────────────────── */
  .week-item {
    display: flex; align-items: center; gap: 20px;
    padding: 14px 20px; border-radius: 8px; margin-bottom: 8px;
    background: #FAFAFA; border: 1px solid #E5E7EB;
  }
  .week-item.critical { background: #FEF2F2; border-color: #FECACA; }
  .week-day {
    font-size: 12px; font-weight: 700; color: #374151;
    font-family: 'JetBrains Mono', monospace; min-width: 72px; flex-shrink: 0;
  }
  .week-event { font-size: 13.5px; color: #1F2937; flex: 1; line-height: 1.4; }
  .week-tag {
    font-size: 9px; font-weight: 700; letter-spacing: 0.06em;
    padding: 3px 10px; border-radius: 3px; text-transform: uppercase; flex-shrink: 0;
  }

  /* ── Pattern cards ───────────────────────────────────────── */
  .pattern {
    padding: 22px 26px; border-radius: 10px; cursor: pointer;
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155; margin-bottom: 16px;
    transition: border-color 0.2s ease; overflow: hidden;
  }
  .pattern:hover { border-color: #475569; }
  .pattern.expanded { border-color: #64748B; }
  .pattern-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 16px;
  }
  .pattern-label {
    font-size: 9px; font-weight: 700; letter-spacing: 0.12em;
    color: #94A3B8; margin-bottom: 8px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
  }
  .pattern h3 {
    font-size: 18px; font-weight: 700; margin: 0 0 6px;
    font-family: 'Source Serif 4', Georgia, serif; color: #F1F5F9;
  }
  .pattern .subtitle {
    font-size: 13.5px; color: #94A3B8; font-weight: 500;
    margin-bottom: 10px; line-height: 1.5;
  }
  .pattern-cats { display: flex; gap: 6px; flex-wrap: wrap; }
  .pattern-cat {
    font-size: 9px; font-weight: 600; letter-spacing: 0.04em;
    color: #CBD5E1; background: #334155; padding: 2px 8px; border-radius: 3px;
  }
  .pattern .chevron { color: #64748B; flex-shrink: 0; }
  .pattern.expanded .chevron { transform: rotate(180deg); }
  .pattern-detail {
    max-height: 0; overflow: hidden; transition: max-height 0.5s ease;
  }
  .pattern.expanded .pattern-detail { max-height: 2000px; }
  .pattern-detail-inner {
    margin-top: 18px; border-top: 1px solid #1E293B; padding-top: 18px;
  }
  .pattern-detail p {
    font-size: 14px; color: #CBD5E1; line-height: 1.8; margin-bottom: 16px;
  }
  .pattern-detail p:last-child { margin-bottom: 4px; }

  /* ── Deep Dive ───────────────────────────────────────────── */
  .deep-dive { padding: 4px 0 0; }
  .deep-dive-kicker {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    color: #6366F1; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 16px;
  }
  .deep-dive-kicker-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #6366F1;
  }
  .deep-dive h2 {
    font-size: 26px; font-weight: 700; color: #111827; line-height: 1.25;
    font-family: 'Source Serif 4', Georgia, serif; margin-bottom: 14px;
  }
  .deep-dive .standfirst {
    font-size: 16px; color: #374151; line-height: 1.6; font-weight: 500;
    margin-bottom: 24px; padding-bottom: 24px;
    border-bottom: 1px solid #E5E7EB;
  }
  .deep-dive-why {
    font-size: 12.5px; color: #6B7280; font-style: italic;
    margin-bottom: 28px; padding: 12px 16px;
    background: #F5F3FF; border-left: 3px solid #6366F1;
    border-radius: 0 6px 6px 0; line-height: 1.5;
  }
  .deep-dive .body p {
    font-size: 15.5px; color: #1F2937; line-height: 1.85; margin-bottom: 22px;
    font-family: 'Source Serif 4', Georgia, serif;
  }
  .deep-dive .body p:last-child { margin-bottom: 0; }

  /* ── Signal Scan ─────────────────────────────────────────── */
  .sig-card {
    padding: 14px 18px; border-radius: 8px; margin-bottom: 8px;
    background: #FAFAFA; border: 1px solid #E5E7EB;
    transition: border-color 0.15s ease;
  }
  .sig-card:not(.sig-quiet):hover { border-color: #D1D5DB; }
  .sig-quiet { opacity: 0.45; }
  .sig-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 7px;
  }
  .sig-left { display: flex; align-items: center; gap: 10px; }
  .sig-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .sig-strong { background: #10B981; }
  .sig-light  { background: #F59E0B; }
  .sig-none   { background: #D1D5DB; }
  .sig-name {
    font-size: 13.5px; font-weight: 700; color: #111827; margin-right: 5px;
  }
  .sig-handle {
    font-size: 11px; color: #9CA3AF;
    font-family: 'JetBrains Mono', monospace;
  }
  .sig-source-badge {
    font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
    color: #6B7280; background: #F3F4F6; border: 1px solid #E5E7EB;
    padding: 2px 7px; border-radius: 3px; text-transform: uppercase;
  }
  .sig-title {
    display: block; font-size: 13.5px; font-weight: 600; color: #111827;
    text-decoration: none; margin-bottom: 5px; line-height: 1.35;
    font-family: 'Source Serif 4', Georgia, serif;
  }
  .sig-title:hover { color: #3B82F6; }
  .sig-summary {
    font-size: 13px; color: #374151; line-height: 1.65; margin-bottom: 5px;
  }
  .sig-connection { font-size: 12px; color: #6366F1; font-style: italic; }
  .sig-quiet-section { margin-top: 20px; }
  .sig-quiet-label {
    font-size: 9px; font-weight: 700; letter-spacing: 0.10em;
    color: #D1D5DB; text-transform: uppercase; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Sources ─────────────────────────────────────────────── */
  .story-sources, .deep-dive-sources {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-top: 16px; padding-top: 14px;
    border-top: 1px solid rgba(0,0,0,0.06);
  }
  .deep-dive-sources { margin-top: 28px; padding-top: 22px; border-top: 1px solid #E5E7EB; }
  .sources-label {
    font-size: 9px; font-weight: 700; letter-spacing: 0.10em;
    color: #9CA3AF; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; flex-shrink: 0;
  }
  .sources-list { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .source-link {
    font-size: 12px; color: #3B82F6; text-decoration: none;
    border-bottom: 1px solid #BFDBFE; line-height: 1.3; transition: color 0.15s;
  }
  .source-link:hover { color: #1D4ED8; border-bottom-color: #93C5FD; }
  .source-sep { font-size: 11px; color: #D1D5DB; }
</style>
</head>
<body>

<div class="header">
  <div class="header-label">DAILY INTELLIGENCE BRIEFING</div>
  <h1>{{DATE}}</h1>
  <div class="status-bar">
    <span class="status-urgent"><span class="pulse-dot"></span>&nbsp;{{URGENT_COUNT}} URGENT</span>
    <span class="status-count">{{ACTIVE_COUNT}} of {{TOTAL_SECTORS}} sectors active</span>
  </div>
</div>

<div class="tabs">
  <button class="tab active" onclick="showSection('stories', this)">Top Stories</button>
  <button class="tab" onclick="showSection('sectors', this)">Sector Scan</button>
  <button class="tab" onclick="showSection('week', this)">Week Ahead</button>
  <button class="tab" onclick="showSection('patterns', this)">Pattern Watch</button>
  <button class="tab" onclick="showSection('deepdive', this)">Deep Dive</button>
  <button class="tab" onclick="showSection('signals', this)">Signal Scan</button>
  <a class="arc-nav-link" href="./archive.html">
    Archive
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg>
  </a>
</div>

<div id="stories" class="section active">
  <p class="hint">Click any story to expand the full analysis.</p>
  {{STORIES_HTML}}
</div>

<div id="sectors" class="section">
  <p class="hint">Green dot = real news today. Click to expand.</p>
  {{SECTORS_HTML}}
</div>

<div id="week" class="section">
  {{WEEK_HTML}}
</div>

<div id="patterns" class="section">
  <p class="hint">Speculative analysis connecting dots across sectors. Click to expand.</p>
  {{PATTERNS_HTML}}
</div>

<div id="deepdive" class="section">
  {{DEEP_DIVE_HTML}}
</div>

<div id="signals" class="section">
  <p class="hint">Recent output from tracked voices. Green = substantive new piece. Amber = light post. Grey = quiet.</p>
  {{SIGNAL_SCAN_HTML}}
</div>

<script>
function showSection(id, btn) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
function toggleStory(el) {
  const p = el.dataset.priority;
  if (el.classList.contains('expanded')) {
    el.classList.remove('expanded', 'expanded-urgent', 'expanded-high', 'expanded-tracking');
  } else {
    el.classList.add('expanded', 'expanded-' + p);
  }
}
function toggleSector(el)  { el.classList.toggle('expanded'); }
function togglePattern(el) { el.classList.toggle('expanded'); }
</script>
</body>
</html>"""


# ── HTML builders ─────────────────────────────────────────────────────────────

TAG_COLORS = {
    "critical": "#DC2626",
    "defense":  "#7C3AED",
    "space":    "#0284C7",
    "data":     "#059669",
    "earnings": "#D97706",
    "policy":   "#6366F1",
    "tech":     "#0891B2",
    "other":    "#6B7280",
}


def build_sources_html(sources, css_class="story-sources"):
    if not sources:
        return ""
    links = []
    for i, src in enumerate(sources):
        sep = '<span class="source-sep">·</span>' if i > 0 else ""
        links.append(
            f'{sep}<a class="source-link" href="{src["url"]}" target="_blank" rel="noopener">'
            f'{src["name"]}</a>'
        )
    return (
        f'<div class="{css_class}">'
        f'<span class="sources-label">SOURCES</span>'
        f'<div class="sources-list">{"".join(links)}</div>'
        f'</div>'
    )


def build_stories_html(stories):
    html = ""
    for s in stories:
        p = s["priority"]
        paragraphs = "\n".join(
            f"<p>{para.strip()}</p>"
            for para in s["detail"].split("\n\n")
            if para.strip()
        )
        sources_html = build_sources_html(s.get("sources", []))
        html += f"""
    <div class="story" data-priority="{p}" onclick="toggleStory(this)">
      <div class="story-top">
        <div class="story-dot dot-{p}"></div>
        <div class="story-content">
          <div class="story-meta">
            <span class="priority-tag tag-{p}">{p.upper()}</span>
            <span class="sector-label">{s['sector']}</span>
          </div>
          <h3>{s['headline']}</h3>
          <div class="one-liner">{s['one_liner']}</div>
        </div>
        <div class="chevron">▾</div>
      </div>
      <div class="story-detail">
        <div class="story-detail-inner">{paragraphs}{sources_html}</div>
      </div>
    </div>"""
    return html


def build_sectors_html(sectors):
    html = ""
    for s in sectors:
        dot_class = "active" if s["has_news"] else "quiet"
        summary = (
            f'<div class="sector-summary">{s["summary"]}</div>'
            if s.get("summary")
            else ""
        )
        html += f"""
    <div class="sector-item" onclick="toggleSector(this)">
      <div class="sector-top">
        <div class="sector-dot {dot_class}"></div>
        <span class="sector-name">{s['name']}</span>
      </div>
      {summary}
    </div>"""
    return html


def build_week_html(items):
    html = ""
    for item in items:
        critical = " critical" if item["tag"] == "critical" else ""
        color = TAG_COLORS.get(item["tag"], "#6B7280")
        html += f"""
    <div class="week-item{critical}">
      <span class="week-day">{item['day']}</span>
      <span class="week-event">{item['event']}</span>
      <span class="week-tag" style="color:#fff;background:{color}">{item['tag']}</span>
    </div>"""
    return html


def build_patterns_html(patterns):
    html = ""
    for p in patterns:
        cats = "".join(
            f'<span class="pattern-cat">{c}</span>' for c in p["categories"]
        )
        paragraphs = "\n".join(
            f"<p>{para.strip()}</p>"
            for para in p["content"].split("\n\n")
            if para.strip()
        )
        html += f"""
    <div class="pattern" onclick="togglePattern(this)">
      <div class="pattern-top">
        <div>
          <div class="pattern-label">SPECULATIVE ANALYSIS</div>
          <h3>{p['title']}</h3>
          <div class="subtitle">{p['subtitle']}</div>
          <div class="pattern-cats">{cats}</div>
        </div>
        <div class="chevron">▾</div>
      </div>
      <div class="pattern-detail">
        <div class="pattern-detail-inner">{paragraphs}</div>
      </div>
    </div>"""
    return html


def build_deep_dive_html(deep_dive):
    if not deep_dive or "headline" not in deep_dive:
        return '<p class="hint">Deep dive unavailable for this briefing.</p>'

    paragraphs = "\n".join(
        f"<p>{para.strip()}</p>"
        for para in deep_dive["body"].split("\n\n")
        if para.strip()
    )
    why_today = deep_dive.get("why_today", "")
    why_html = (
        f'<div class="deep-dive-why">{why_today}</div>' if why_today else ""
    )
    sources_html = build_sources_html(deep_dive.get("sources", []), "deep-dive-sources")

    return f"""
  <div class="deep-dive">
    <div class="deep-dive-kicker">
      <span class="deep-dive-kicker-dot"></span>TODAY'S DEEP DIVE
    </div>
    <h2>{deep_dive['headline']}</h2>
    <div class="standfirst">{deep_dive['standfirst']}</div>
    {why_html}
    <div class="body">{paragraphs}</div>
    {sources_html}
  </div>"""


def build_signal_scan_html(signals):
    if not signals:
        return '<p class="hint">Signal scan unavailable for this briefing.</p>'

    active = [s for s in signals if s.get("signal") in ("strong", "light")]
    quiet  = [s for s in signals if s.get("signal") == "none"]

    def card(s, is_quiet=False):
        signal  = s.get("signal", "none")
        name    = s.get("name", "")
        handle  = s.get("handle", "")
        source  = s.get("source", "")
        summary = s.get("summary", "")
        conn    = s.get("connection", "")
        url     = s.get("url", "")
        title   = s.get("latest_title", "")

        dot_cls    = {"strong": "sig-strong", "light": "sig-light"}.get(signal, "sig-none")
        quiet_cls  = " sig-quiet" if is_quiet else ""
        handle_html = f'<span class="sig-handle">{handle}</span>' if handle else ""
        badge_html  = f'<span class="sig-source-badge">{source.upper()}</span>' if source else ""

        if title and url:
            title_html = f'<a class="sig-title" href="{url}" target="_blank" rel="noopener">{title}</a>'
        elif title:
            title_html = f'<div class="sig-title" style="cursor:default">{title}</div>'
        else:
            title_html = ""

        summary_html = f'<div class="sig-summary">{summary}</div>' if summary else ""
        conn_html    = f'<div class="sig-connection">↳ {conn}</div>' if conn else ""

        return f"""
    <div class="sig-card{quiet_cls}">
      <div class="sig-header">
        <div class="sig-left">
          <div class="sig-dot {dot_cls}"></div>
          <div><span class="sig-name">{name}</span>{handle_html}</div>
        </div>
        {badge_html}
      </div>
      {title_html}{summary_html}{conn_html}
    </div>"""

    html = "".join(card(s) for s in active)

    if quiet:
        q_cards = "".join(card(s, is_quiet=True) for s in quiet)
        html += f"""
    <div class="sig-quiet-section">
      <div class="sig-quiet-label">QUIET THIS PERIOD — {len(quiet)} voices</div>
      {q_cards}
    </div>"""

    return html


def build_dashboard(data, date_str, cfg):
    stories_html      = build_stories_html(data["top_stories"])
    sectors_html      = build_sectors_html(data["sectors"])
    week_html         = build_week_html(data["week_ahead"])
    patterns_html     = build_patterns_html(data.get("pattern_watch", []))
    deep_dive_html    = build_deep_dive_html(data.get("deep_dive"))
    signal_scan_html  = build_signal_scan_html(data.get("signal_scan", []))

    urgent_count  = sum(1 for s in data["top_stories"] if s["priority"] == "urgent")
    active_count  = sum(1 for s in data["sectors"] if s["has_news"])
    total_sectors = len(data["sectors"])

    html = HTML_TEMPLATE
    html = html.replace("{{DATE}}",             date_str)
    html = html.replace("{{STORIES_HTML}}",     stories_html)
    html = html.replace("{{SECTORS_HTML}}",     sectors_html)
    html = html.replace("{{WEEK_HTML}}",        week_html)
    html = html.replace("{{PATTERNS_HTML}}",    patterns_html)
    html = html.replace("{{DEEP_DIVE_HTML}}",   deep_dive_html)
    html = html.replace("{{SIGNAL_SCAN_HTML}}", signal_scan_html)
    html = html.replace("{{URGENT_COUNT}}",     str(urgent_count))
    html = html.replace("{{ACTIVE_COUNT}}",     str(active_count))
    html = html.replace("{{TOTAL_SECTORS}}",    str(total_sectors))
    return html


# ── JSON extraction ────────────────────────────────────────────────────────────

def extract_json(text):
    """Robustly extract a JSON object or array from Claude's response."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { or [ and balance from there
    for open_c, close_c in [('{', '}'), ('[', ']')]:
        start = text.find(open_c)
        if start == -1:
            continue

        depth = 0
        end = -1
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue

    return None


# ── Wildcard history ───────────────────────────────────────────────────────────

def load_wildcard_history(output_dir):
    path = os.path.join(output_dir, "wildcard_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_wildcard_history(output_dir, history, new_topic):
    path = os.path.join(output_dir, "wildcard_history.json")
    history = list(history)  # don't mutate the caller's list
    history.append({"date": datetime.date.today().isoformat(), "topic": new_topic})
    history = history[-30:]  # keep 30 days
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return history


# ── API calls ──────────────────────────────────────────────────────────────────

def _retry(fn, label, retries=MAX_RETRIES):
    """Run fn() up to retries+1 times with exponential back-off."""
    for attempt in range(retries + 1):
        try:
            return fn(attempt)
        except anthropic.RateLimitError:
            wait = 65 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s...")
            if attempt < retries:
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"  {label} error (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                time.sleep(wait)
            else:
                raise


def generate_briefing_data(cfg, output_dir):
    print("  Calling Claude Sonnet (web search) — main briefing...")
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    system = build_system_prompt(cfg) + "\n\n" + build_format_instructions(cfg)

    def attempt(n):
        full_text = run_model(
            label="main briefing",
            model="claude-sonnet-4-6",
            prompt=f"Go. Today is {today}.",
            system=system,
            max_tokens=8000,
            use_search=True,
            cfg=cfg,
            search_kind="main",
            max_turns=8,
        )

        if not full_text.strip():
            raise ValueError("Empty response from API")

        data = extract_json(full_text)
        if data is None:
            debug = os.path.join(output_dir, f"debug-main-{n}.txt")
            with open(debug, "w") as f:
                f.write(full_text)
            raise ValueError(f"No valid JSON found (debug saved to {debug})")

        missing = [k for k in ("top_stories", "sectors", "week_ahead") if k not in data]
        if missing:
            raise ValueError(f"Missing keys in response: {missing}")

        print(
            f"  ✓ {len(data['top_stories'])} stories, "
            f"{len(data['sectors'])} sectors, "
            f"{len(data['week_ahead'])} events"
        )
        return data

    return _retry(attempt, "main briefing")


def generate_deep_dive(cfg, output_dir):
    print("  Calling Claude Sonnet (web search) — deep dive...")
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    history = load_wildcard_history(output_dir)
    prompt = build_deep_dive_prompt(cfg, history) + f"\n\nToday is {today}."

    def attempt(n):
        full_text = run_model(
            label="deep dive",
            model="claude-sonnet-4-6",
            prompt=prompt,
            max_tokens=3000,
            use_search=True,
            cfg=cfg,
            search_kind="deep_dive",
            max_turns=6,
        )
        data = extract_json(full_text)

        if data is None or "headline" not in data:
            raise ValueError("Deep dive returned no valid JSON with 'headline' key")

        save_wildcard_history(output_dir, history, data.get("topic", data["headline"]))
        print(f"  ✓ Deep dive: {data.get('topic', data['headline'])}")
        return data

    try:
        return _retry(attempt, "deep dive")
    except Exception as e:
        print(f"  Deep dive failed after all retries: {e}")
        return None


def generate_patterns(briefing_data):
    print("  Calling Claude Haiku — pattern analysis...")
    payload = json.dumps({
        "top_stories": briefing_data["top_stories"],
        "sectors": briefing_data["sectors"],
    })

    def attempt(n):
        raw = run_model(
            label="pattern watch",
            model="claude-haiku-4-5-20251001",
            prompt=PATTERN_PROMPT_BASE + payload,
            max_tokens=2500,
            use_search=False,
            cfg=None,
            max_turns=1,
        ).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
        patterns = json.loads(raw)
        print(f"  ✓ {len(patterns)} pattern analyses")
        return patterns

    try:
        return _retry(attempt, "pattern watch")
    except Exception as e:
        print(f"  Pattern watch failed: {e}")
        return [{
            "title": "Pattern analysis unavailable",
            "subtitle": "Failed to generate — check logs",
            "categories": ["Error"],
            "content": str(e),
        }]


# ── Signal Scan ───────────────────────────────────────────────────────────────

def load_signal_scan_cache(output_dir):
    path = os.path.join(output_dir, "signal_scan_cache.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cache = json.load(f)
        cached_date = datetime.date.fromisoformat(cache.get("date", "2000-01-01"))
        age_days = (datetime.date.today() - cached_date).days
        if age_days < 7:
            print(f"  Using cached signal scan from {cached_date} ({age_days}d ago)")
            return cache.get("signals", [])
    except Exception:
        pass
    return None


def save_signal_scan_cache(output_dir, signals):
    path = os.path.join(output_dir, "signal_scan_cache.json")
    with open(path, "w") as f:
        json.dump({"date": datetime.date.today().isoformat(), "signals": signals}, f, indent=2)


def fetch_rss_feeds(voices):
    """Fetch recent items from RSS feeds. Returns {name: [items]}."""
    if not FEEDPARSER_AVAILABLE:
        print("  feedparser not installed — skipping RSS fetch. Run: pip install feedparser")
        return {v["name"]: [] for v in voices}

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    results = {}

    for voice in voices:
        name    = voice["name"]
        rss_url = voice.get("rss")
        if not rss_url:
            results[name] = []
            continue
        try:
            feed = feedparser.parse(rss_url, agent="MorningBriefing/1.0")
            items = []
            for entry in feed.entries[:5]:
                pub = None
                for attr in ("published_parsed", "updated_parsed"):
                    val = getattr(entry, attr, None)
                    if val:
                        try:
                            pub = datetime.datetime(*val[:6], tzinfo=datetime.timezone.utc)
                        except Exception:
                            pass
                        break

                if pub is None or pub > cutoff:
                    summary = getattr(entry, "summary", "") or ""
                    # Strip HTML tags from summary
                    summary = re.sub(r"<[^>]+>", " ", summary)
                    summary = re.sub(r"\s+", " ", summary).strip()[:400]
                    items.append({
                        "title":     getattr(entry, "title", "").strip(),
                        "url":       getattr(entry, "link",  "").strip(),
                        "published": pub.strftime("%Y-%m-%d") if pub else "",
                        "summary":   summary,
                    })
            results[name] = items
            if items:
                print(f"    {name}: {len(items)} RSS item(s)")
        except Exception as e:
            print(f"    RSS fetch failed for {name}: {e}")
            results[name] = []

    return results


def generate_signal_scan(voices, rss_content, briefing_data, cfg):
    """One Sonnet + web_search call: process RSS + search X for handles without RSS."""
    print("  Calling Claude Sonnet (web search) — signal scan...")
    today  = datetime.date.today().strftime("%A, %B %d, %Y")

    # Build RSS context and identify voices needing X search
    rss_context     = []
    search_targets  = []

    today_iso = datetime.date.today().isoformat()

    for voice in voices:
        name     = voice["name"]
        handle   = voice.get("x_handle")
        items    = rss_content.get(name, [])
        custom_q = voice.get("search_query")

        if items:
            block = f"### {name}" + (f" (@{handle})" if handle else "") + "\n"
            for it in items[:3]:
                block += f"- [{it['title']}]({it['url']})"
                if it["published"]:
                    block += f" — {it['published']}"
                block += "\n"
                if it["summary"]:
                    block += f"  {it['summary'][:300]}…\n"
            rss_context.append(block)

            # Only search X if RSS content is not from today — skip if already fresh
            rss_fresh = any(it.get("published", "") == today_iso for it in items)
            if not rss_fresh:
                if custom_q:
                    search_targets.append(f"{name}: search [{custom_q}]")
                elif handle:
                    search_targets.append(f"{name} (@{handle}): search recent X/Twitter posts")
        else:
            # No RSS at all — always search
            if custom_q:
                search_targets.append(f"{name}: search [{custom_q}]")
            elif handle:
                search_targets.append(f"{name} (@{handle}): search recent X/Twitter posts")

    rss_block    = "\n".join(rss_context) or "No RSS content retrieved."
    search_block = "\n".join(search_targets) or "None."
    topics_str   = ", ".join(t["name"] for t in cfg["topics"])
    headlines    = " | ".join(s.get("headline", "") for s in briefing_data.get("top_stories", []))
    all_names    = [v["name"] for v in voices]

    prompt = f"""Today is {today}.

Review recent output from a curated list of thinkers and analysts, then return a signal scan.

TODAY'S TOP STORIES: {headlines}
TODAY'S TOPIC AREAS: {topics_str}

RSS CONTENT RETRIEVED (last 7 days):
{rss_block}

VOICES TO SEARCH ON X / WEB (search ALL of these — combine with any RSS content above):
{search_block}

For each voice, combine their RSS content (if available above) with their most recent X posts, then return a JSON array covering ALL {len(all_names)} voices in this exact order: {', '.join(all_names)}

[
  {{
    "name": "Exact name as listed above",
    "handle": "@handle or empty string",
    "source": "substack" | "newsletter" | "x" | "essay" | "blog" | "mixed",
    "signal": "strong" | "light" | "none",
    "latest_title": "Title of most recent piece or post, or empty string",
    "url": "Direct URL, or empty string",
    "summary": "1-2 sentences on what they wrote or said. Empty string if nothing in 7 days.",
    "connection": "How this connects to today's stories/topics. Empty string if none."
  }}
]

Signal levels: "strong" = substantive piece or thread (>200 words). "light" = brief post or short take. "none" = nothing in the last 7 days.
Skip purely promotional, congratulatory, or off-topic content.
IMPORTANT: Make at most 1 web search per voice. Do not search exhaustively.
Return ONLY the JSON array. No markdown, no preamble."""

    def attempt(n):
        full_text = run_model(
            label="signal scan",
            model="claude-sonnet-4-6",
            prompt=prompt,
            max_tokens=6000,
            use_search=True,
            cfg=cfg,
            search_kind="signal",
            max_turns=12,
        )
        data = extract_json(full_text)
        if data is None or not isinstance(data, list):
            # Log the raw response tail for debugging
            print(f"  Raw response tail: {full_text[-300:]}")
            raise ValueError("Signal scan returned no valid JSON array")
        active = sum(1 for s in data if s.get("signal") in ("strong", "light"))
        print(f"  ✓ {len(data)} voices scanned, {active} with recent signal")
        return data

    try:
        return _retry(attempt, "signal scan")
    except Exception as e:
        print(f"  Signal scan failed: {e}")
        return []


# ── Archive ───────────────────────────────────────────────────────────────────

def update_archive(output_dir):
    """Scan output_dir for all briefing-*.json files and regenerate archive.html."""
    import glob as _glob

    json_files = sorted(
        _glob.glob(os.path.join(output_dir, "briefing-*.json")),
        reverse=True,  # newest first
    )

    entries = []
    for jf in json_files:
        date_str = os.path.basename(jf).replace("briefing-", "").replace(".json", "")
        html_file = f"briefing-{date_str}.html"
        if not os.path.exists(os.path.join(output_dir, html_file)):
            continue
        try:
            date_obj = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        try:
            with open(jf) as f:
                data = json.load(f)
            urgent  = sum(1 for s in data.get("top_stories", []) if s.get("priority") == "urgent")
            active  = sum(1 for s in data.get("sectors", []) if s.get("has_news"))
            n_stories = len(data.get("top_stories", []))
            deep_dive_topic = data.get("deep_dive", {}) or {}
            deep_dive_topic = deep_dive_topic.get("topic", "")
        except Exception:
            urgent = active = n_stories = 0
            deep_dive_topic = ""

        entries.append({
            "date_str":    date_str,
            "date_fmt":    date_obj.strftime("%A, %B %d, %Y"),
            "html_file":   html_file,
            "urgent":      urgent,
            "active":      active,
            "n_stories":   n_stories,
            "deep_dive":   deep_dive_topic,
        })

    today_str = datetime.date.today().isoformat()
    rows = ""
    for e in entries:
        is_today = e["date_str"] == today_str
        today_class = " today" if is_today else ""
        today_badge = '<span class="arc-today-badge">TODAY</span>' if is_today else ""
        urgent_badge = (
            f'<span class="arc-urgent">{e["urgent"]} URGENT</span>'
            if e["urgent"] else ""
        )
        dive_snippet = (
            f'<div class="arc-dive">↳ {e["deep_dive"]}</div>'
            if e["deep_dive"] else ""
        )
        rows += f"""
    <a class="arc-entry{today_class}" href="./{e['html_file']}">
      <div class="arc-date">{e['date_fmt']}</div>
      <div class="arc-meta">
        {today_badge}
        {urgent_badge}
        <span class="arc-stat">{e['active']} sectors active</span>
        <span class="arc-stat">{e['n_stories']} stories</span>
      </div>
      {dive_snippet}
    </a>"""

    today_file = f"briefing-{datetime.date.today().strftime('%Y-%m-%d')}.html"
    archive_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Briefing Archive</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'DM Sans', system-ui, sans-serif;
    max-width: 720px; margin: 0 auto; padding: 0 24px 80px;
    color: #111827; background: #fff; -webkit-font-smoothing: antialiased;
  }}
  .top-bar {{
    padding-top: 28px; margin-bottom: 32px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .back-link {{
    font-size: 13px; color: #6B7280; text-decoration: none;
    display: inline-flex; align-items: center; gap: 5px;
    transition: color 0.15s ease;
  }}
  .back-link:hover {{ color: #111827; }}
  .header-label {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    color: #9CA3AF; text-transform: uppercase; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }}
  h1 {{
    font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 6px;
    font-family: 'Source Serif 4', Georgia, serif;
  }}
  .header-count {{ font-size: 13px; color: #6B7280; margin-bottom: 28px; }}
  .arc-entry {{
    display: block; text-decoration: none; color: inherit;
    padding: 16px 20px; border-radius: 8px; margin-bottom: 8px;
    background: #FAFAFA; border: 1px solid #E5E7EB;
    transition: border-color 0.2s ease, background 0.2s ease;
  }}
  .arc-entry:hover {{ border-color: #9CA3AF; background: #F3F4F6; }}
  .arc-entry.today {{ border-color: #111827; background: #F9FAFB; }}
  .arc-today-badge {{
    font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
    color: #111827; background: #F3F4F6; border: 1px solid #D1D5DB;
    padding: 2px 8px; border-radius: 3px; text-transform: uppercase;
  }}
  .arc-date {{
    font-size: 15px; font-weight: 600; color: #111827; margin-bottom: 6px;
    font-family: 'Source Serif 4', Georgia, serif;
  }}
  .arc-meta {{
    display: flex; align-items: center; gap: 8px;
    flex-wrap: wrap; margin-bottom: 4px;
  }}
  .arc-urgent {{
    font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
    color: #991B1B; background: #FEE2E2; border: 1px solid #FECACA;
    padding: 2px 8px; border-radius: 3px; text-transform: uppercase;
  }}
  .arc-stat {{ font-size: 12px; color: #6B7280; }}
  .arc-dive {{ font-size: 12px; color: #6366F1; font-style: italic; margin-top: 2px; }}
</style>
</head>
<body>
<div class="top-bar">
  <a class="back-link" href="./{today_file}">
    ← Today's briefing
  </a>
</div>
<div class="header-label">MORNING BRIEFING</div>
<h1>Archive</h1>
<div class="header-count">{len(entries)} briefing{"s" if len(entries) != 1 else ""}</div>
{rows}
</body>
</html>"""

    archive_path = os.path.join(output_dir, "archive.html")
    with open(archive_path, "w") as f:
        f.write(archive_html)
    print(f"  archive.html updated ({len(entries)} entries)")


# ── Index redirect ────────────────────────────────────────────────────────────

def update_index(html_path, output_dir):
    filename = os.path.basename(html_path)
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url=./{filename}">
  <title>Redirecting...</title>
</head>
<body><p>Redirecting to <a href="./{filename}">today's briefing</a>...</p></body>
</html>""")
    print(f"  index.html → {filename}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Morning Briefing Dashboard Generator")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if today's briefing already exists"
    )
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY and not (BRIEFING_BACKEND == "pro" and CLAUDE_CODE_OAUTH_TOKEN):
        print("Error: set ANTHROPIC_API_KEY (or BRIEFING_BACKEND=pro with CLAUDE_CODE_OAUTH_TOKEN)")
        sys.exit(1)
    if BRIEFING_BACKEND == "pro" and not ANTHROPIC_API_KEY:
        print("Note: Pro backend set without ANTHROPIC_API_KEY — no API fallback if a call fails.")

    cfg = load_config(args.config)
    output_dir = os.path.expanduser(
        os.environ.get(
            "BRIEFING_OUTPUT_DIR",
            cfg.get("output", {}).get("dir", "~/briefings"),
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    today     = datetime.date.today()
    date_str  = today.strftime("%A, %B %d, %Y")
    file_date = today.strftime("%Y-%m-%d")
    html_path = os.path.join(output_dir, f"briefing-{file_date}.html")
    json_path = os.path.join(output_dir, f"briefing-{file_date}.json")

    if os.path.exists(html_path) and not args.force:
        print(f"Already generated for {file_date}. Use --force to regenerate.")
        return

    sections = cfg.get("sections", DEFAULT_CONFIG["sections"])

    start = time.time()
    print(f"=== Morning Briefing — {date_str} ===\n")

    voices     = cfg.get("voices", DEFAULT_CONFIG["voices"])
    do_dive    = sections.get("deep_dive",    True) and _deep_dive_due(cfg)
    if sections.get("deep_dive", True) and not _deep_dive_due(cfg):
        print("  Deep dive skipped today (economy: alternate-day schedule)")
    do_pattern = sections.get("pattern_watch",True)
    frequency  = cfg.get("signal_scan_frequency", "weekly")

    # Check signal scan cache for weekly mode
    cached_signals = None
    if sections.get("signal_scan", True) and voices and frequency == "weekly" and not args.force:
        cached_signals = load_signal_scan_cache(output_dir)

    do_signal = sections.get("signal_scan", True) and bool(voices) and (cached_signals is None)

    # ── Stage 1: independent calls — run in parallel ──────────────────────────
    print("[1/3] Stage 1 (parallel): main briefing + deep dive + RSS fetch...")

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_main = ex.submit(generate_briefing_data, cfg, output_dir)
        f_dive = ex.submit(generate_deep_dive, cfg, output_dir) if do_dive else None
        f_rss  = ex.submit(fetch_rss_feeds, voices)              if do_signal else None

        data        = f_main.result()
        deep_dive   = f_dive.result() if f_dive else None
        rss_content = f_rss.result()  if f_rss  else {}

    data["deep_dive"] = deep_dive

    # ── Stage 2: both need `data` — run in parallel ───────────────────────────
    print("[2/3] Stage 2 (parallel): signal scan + pattern watch...")

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_sig = ex.submit(generate_signal_scan, voices, rss_content, data, cfg) if do_signal  else None
        f_pat = ex.submit(generate_patterns, data)                               if do_pattern else None

        fresh_signals         = f_sig.result() if f_sig else None
        data["pattern_watch"] = f_pat.result() if f_pat else []

    if fresh_signals is not None:
        data["signal_scan"] = fresh_signals
        if frequency == "weekly":
            save_signal_scan_cache(output_dir, fresh_signals)
            print(f"  Signal scan cached — next fresh run in 7 days")
    elif cached_signals is not None:
        data["signal_scan"] = cached_signals
    else:
        data["signal_scan"] = []

    # ── Stage 3: save and build ───────────────────────────────────────────────
    print("[3/3] Saving and building dashboard...")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    html = build_dashboard(data, date_str, cfg)
    with open(html_path, "w") as f:
        f.write(html)
    update_index(html_path, output_dir)
    update_archive(output_dir)

    elapsed = time.time() - start
    print_usage_summary()
    print(f"\n✓ Done in {elapsed:.0f}s")
    print(f"  File:    {html_path}")
    print(f"  Data:    {json_path}")
    print(f"  Archive: {os.path.join(output_dir, 'archive.html')}")
    print(f"  Local:   open {html_path}")


if __name__ == "__main__":
    main()

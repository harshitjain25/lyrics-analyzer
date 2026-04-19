"""Reusable Plotly chart factory for the Song Lyrics Analyzer dashboard."""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PALETTE = [
    "#7B2FBE", "#4361EE", "#F77F00", "#E63946", "#2EC4B6",
    "#FF9F1C", "#6A0572", "#3A86FF", "#FB5607", "#8338EC",
]

EMOTION_COLORS = {
    "joy": "#F77F00", "sadness": "#4361EE", "anger": "#E63946",
    "fear": "#7B2FBE", "surprise": "#2EC4B6", "disgust": "#2DC653",
    "neutral": "#ADB5BD",
}


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", size=13),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title,
                 color_discrete_sequence=PALETTE, barmode="stack")
    return apply_theme(fig)


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title,
                  color_discrete_sequence=PALETTE, markers=True)
    return apply_theme(fig)


def radar_chart(categories: list[str], values_dict: dict[str, list[float]], title: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, values) in enumerate(values_dict.items()):
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.7,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=title)
    return apply_theme(fig)


def heatmap(df: pd.DataFrame, x: str, y: str, values: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index=y, columns=x, values=values, aggfunc="mean").fillna(0)
    fig = px.imshow(pivot, title=title, color_continuous_scale="Purples", aspect="auto")
    return apply_theme(fig)


def horizontal_bar(df: pd.DataFrame, x: str, y: str, color_col: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color_col, title=title,
                 orientation="h", color_discrete_sequence=PALETTE)
    return apply_theme(fig)


def emotion_bar(emotion_scores: dict[str, float]) -> go.Figure:
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    colors = [EMOTION_COLORS.get(e, "#ADB5BD") for e in emotions]
    fig = go.Figure(go.Bar(
        x=scores, y=emotions, orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(title="Emotion Scores", xaxis_range=[0, 1])
    return apply_theme(fig)

"""Sidebar filter widgets and DataFrame filtering helpers."""
from __future__ import annotations
import streamlit as st
import pandas as pd

GENRES = ["pop", "rock", "hip-hop", "country", "r&b", "electronic"]
DECADES = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]


def genre_filter(default_all: bool = True) -> list[str]:
    return st.sidebar.multiselect(
        "Genre", GENRES, default=GENRES if default_all else []
    )


def decade_filter(default_all: bool = True) -> list[str]:
    return st.sidebar.multiselect(
        "Decade", DECADES, default=DECADES if default_all else []
    )


def get_filtered_df(
    songs_df: pd.DataFrame,
    song_topics_df: pd.DataFrame,
    sentiments_df: pd.DataFrame,
) -> pd.DataFrame:
    genres = st.session_state.get("selected_genres", GENRES)
    decades = st.session_state.get("selected_decades", DECADES)

    df = songs_df.copy()
    if genres:
        df = df[df["genre"].isin(genres)]
    if decades:
        df = df[df["decade"].isin(decades)]

    if not song_topics_df.empty:
        df = df.merge(song_topics_df, left_on="id", right_on="song_id", how="left")
    if not sentiments_df.empty:
        sent = sentiments_df.rename(columns={"song_id": "id"})
        df = df.merge(sent, on="id", how="left")

    return df.reset_index(drop=True)

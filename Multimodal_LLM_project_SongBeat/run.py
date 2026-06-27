"""CLI entrypoint for SongBeat.

Examples
--------
    python run.py --audio data/samples/song.mp3 --lyrics data/samples/song.txt
    python run.py --lyrics-text "I'm walking on sunshine" --title "demo"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.graph import analyze_song


def _read_lyrics(args) -> str:
    if args.lyrics_text:
        return args.lyrics_text
    if args.lyrics:
        return Path(args.lyrics).read_text(encoding="utf-8")
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description="Audio-vs-lyric emotional alignment.")
    p.add_argument("--audio", help="Path to an audio file (mp3/wav/...).")
    p.add_argument("--lyrics", help="Path to a .txt file of lyrics.")
    p.add_argument("--lyrics-text", help="Lyrics passed inline as a string.")
    p.add_argument("--title", default="", help="Optional song title.")
    p.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = p.parse_args()

    result = analyze_song(args.audio, _read_lyrics(args), title=args.title)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\nSong: {args.title or '(untitled)'}")
        print(f"  Audio  V/A : {result.audio.valence:+.2f} / {result.audio.arousal:+.2f}")
        print(f"  Lyric  V/A : {result.lyric.valence:+.2f} / {result.lyric.arousal:+.2f} "
              f"({result.lyric.label}, {result.lyric.backend})")
        print(f"  Alignment  : {result.score:.2f}  ->  {result.label}")
        print(f"  Verdict    : {result.explanation}")
        if result.retrieved_context:
            print("  Context    : " +
                  ", ".join(f"{c.get('title','?')}({c.get('similarity','?')})"
                            for c in result.retrieved_context))


if __name__ == "__main__":
    main()

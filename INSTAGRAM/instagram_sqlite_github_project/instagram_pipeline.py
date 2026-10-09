from pathlib import Path
import sqlite3
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "instagram details"
DB_FILE = BASE_DIR / "insta_lite.db"

DATE_COLUMNS = ["created_dat", "created_date", "created_time", "created_times"]

def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files from the expected data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}. Add the source CSV files there."
        )
    frames = {}
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.replace(" ", "_").str.lower()
        frames[path.stem.lower()] = frame
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return frames

def require_frames(frames: dict[str, pd.DataFrame], names: list[str]) -> None:
    missing = [name for name in names if name not in frames]
    if missing:
        raise ValueError(
            "Missing expected CSV files (filename stems): "
            + ", ".join(missing)
        )

def normalize_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize yes/no values and parse recognized date columns."""
    for frame in frames.values():
        for col in frame.columns:
            if frame[col].dropna().isin(["yes", "no"]).all() and frame[col].notna().any():
                frame[col] = frame[col].map({"yes": True, "no": False})
        for col in DATE_COLUMNS:
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce", dayfirst=True)
    return frames

def prepare_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    require_frames(frames, ["users", "photos", "tags", "comments", "likes", "follows"])
    frames = normalize_frames(frames)

    # Add deterministic row-based IDs only when absent.
    for table, id_col in [("likes", "like_id"), ("follows", "follow_id")]:
        if id_col not in frames[table].columns:
            frames[table][id_col] = range(1, len(frames[table]) + 1)

    rename_map = {
        "photos": {"created_dat": "created_date", "id": "photo_id"},
        "users": {"id": "user_id", "private/public": "private"},
        "tags": {"id": "tag_id"},
        "comments": {"id": "comment_id"},
        "likes": {"user_": "user_id", "photo": "photo_id"},
        "follows": {"follower": "follower_user_id", "followee": "user_id"},
    }
    for table, mapping in rename_map.items():
        frames[table] = frames[table].rename(columns=mapping)

    likes = frames["likes"].copy()
    comments = frames["comments"].copy()
    follows = frames["follows"].copy()
    for frame, kind in [(likes, "like"), (comments, "comment"), (follows, "follow")]:
        frame["interaction_type"] = kind
        if "created_time" not in frame.columns:
            frame["created_time"] = pd.NaT

    # Align shared fields before concatenating to make the fact table predictable.
    parts = []
    for frame, kind in [(likes, "like"), (comments, "comment"), (follows, "follow")]:
        part = pd.DataFrame(index=frame.index)
        part["interaction_type"] = kind
        part["interaction_date"] = frame["created_time"]
        part["user_id"] = frame["user_id"] if "user_id" in frame else pd.NA
        part["photo_id"] = frame["photo_id"] if "photo_id" in frame else pd.NA
        part["tag_id"] = frame["tag_id"] if "tag_id" in frame else pd.NA
        for id_col in ["comment_id", "like_id", "follow_id"]:
            part[id_col] = frame[id_col] if id_col in frame else pd.NA
        parts.append(part.reset_index(drop=True))

    interactions = pd.concat(parts, ignore_index=True)
    interactions.insert(0, "interaction_id", range(1, len(interactions) + 1))
    for col in ["user_id", "photo_id", "tag_id", "comment_id", "like_id", "follow_id"]:
        interactions[col] = pd.to_numeric(interactions[col], errors="coerce").astype("Int64")
    frames["interactions"] = interactions
    return frames

def write_database(frames: dict[str, pd.DataFrame], db_file: Path = DB_FILE) -> None:
    """Write normalized frames to SQLite. DataFrame schemas are inferred by pandas."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        for table in ["users", "photos", "tags", "comments", "likes", "follows", "interactions"]:
            if table in frames:
                frames[table].to_sql(table, conn, if_exists="replace", index=False)

def print_summary(db_file: Path = DB_FILE) -> dict[str, int]:
    queries = {
        "photos": "SELECT COUNT(*) FROM photos",
        "users": "SELECT COUNT(*) FROM users",
        "tags": "SELECT COUNT(DISTINCT tag_id) FROM tags",
        "likes": "SELECT COUNT(*) FROM interactions WHERE interaction_type = 'like'",
        "comments": "SELECT COUNT(*) FROM interactions WHERE interaction_type = 'comment'",
        "follows": "SELECT COUNT(*) FROM interactions WHERE interaction_type = 'follow'",
    }
    results = {}
    with sqlite3.connect(str(db_file)) as conn:
        for label, query in queries.items():
            results[label] = int(conn.execute(query).fetchone()[0])
    for label, count in results.items():
        print(f"{label.title()} count: {count}")
    return results

def main() -> None:
    frames = prepare_frames(load_csvs(DATA_DIR))
    write_database(frames)
    print_summary()

if __name__ == "__main__":
    main()

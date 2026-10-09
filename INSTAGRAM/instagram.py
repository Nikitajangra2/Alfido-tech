from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import warnings

# Suppress all pandas date parsing warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------------------------------------------
# 1. Read in and clean data
# ---------------------------------------------------------

data_dir = Path(__file__).parent / 'instagram details'
dfs = {}

try:
    for file_path in data_dir.glob('*.csv'):
        if file_path.is_file():
            file_name = file_path.stem
            df = pd.read_csv(file_path)
            dfs[file_name] = df
except Exception:
    pass

for key, df in dfs.items():
    globals()[f'df_{key}'] = df

# Clean column headers
for key, df in dfs.items():
    df.columns = df.columns.str.replace(' ', '_').str.lower()

# Convert yes/no columns to booleans
for key, df in dfs.items():
    for col in df.columns:
        if df[col].isin(['yes', 'no']).all():
            df[col] = df[col].map({'yes': True, 'no': False})

# Standardize date columns
date_time_cols = ['created_dat', 'created_date', 'created_time', 'created_times']
for key, df in dfs.items():
    for col in date_time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

df_likes['like_id'] = (df_likes.index + 1).astype(int)
df_follows['follow_id'] = (df_follows.index + 1).astype(int)

# Rename columns
df_photos.rename(columns={'created_dat': 'created_date', 'id': 'photo_id'}, inplace=True)
df_users.rename(columns={'id': 'user_id', 'private/public': 'private'}, inplace=True)
df_tags.rename(columns={'id': 'tag_id'}, inplace=True)
df_comments.rename(columns={'user_id': 'user_id', 'id': 'comment_id'}, inplace=True)
df_likes.rename(columns={'user_': 'user_id', 'photo': 'photo_id'}, inplace=True)
df_follows.rename(columns={'follower': 'follower_user_id', 'followee': 'user_id'}, inplace=True)


# ---------------------------------------------------------
# 2. Build Interactions DataFrame (Fact Table)
# ---------------------------------------------------------

df_interactions = pd.concat([df_likes, df_comments, df_follows], ignore_index=True)
df_interactions['interaction_id'] = df_interactions.index + 1

df_interactions['interaction_type'] = (
    ['like'] * len(df_likes) + 
    ['comment'] * len(df_comments) + 
    ['follow'] * len(df_follows)
)

df_interactions['interaction_date'] = df_interactions['created_time']

keep_cols = [
    'interaction_id', 'interaction_type', 'interaction_date', 
    'user_id', 'photo_id', 'tag_id', 'comment_id', 'like_id', 'follow_id'
]
df_interactions = df_interactions[[col for col in df_interactions.columns if col in keep_cols]]

id_cols = ['user_id', 'photo_id', 'comment_id', 'like_id', 'follow_id']
for col in id_cols:
    if col in df_interactions.columns:
        df_interactions[col] = df_interactions[col].astype('Int64')


# ---------------------------------------------------------
# 3. SQLite Database Creation & Population
# ---------------------------------------------------------

db_file = Path(__file__).parent / 'insta_lite.db'
db_file.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(db_file))
cursor = conn.cursor()

# Users Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT,
    created_time DATE,
    private BOOLEAN,
    post_count INTEGER,
    verified_status BOOLEAN
)
''')
df_users.to_sql('users', conn, if_exists='replace', index=False)

# Photos Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS photos (
    photo_id INTEGER PRIMARY KEY,
    image_link TEXT,
    user_id INTEGER,
    created_date DATE,
    insta_filter_used BOOLEAN,
    photo_type TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
)
''')
df_photos.to_sql('photos', conn, if_exists='replace', index=False)

# Tags Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY,
    tag_text TEXT,
    created_time DATE,
    location TEXT
)
''')
df_tags.to_sql('tags', conn, if_exists='replace', index=False)

# Comments Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS comments (
    comment_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    photo_id INTEGER,
    created_time DATETIME,
    posted_date TEXT,
    comment TEXT,
    emoji_used BOOLEAN,
    hashtags_used_count INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (photo_id) REFERENCES photos (photo_id)
)
''')
df_comments.to_sql('comments', conn, if_exists='replace', index=False)

# Likes Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS likes (
    like_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    photo_id INTEGER,
    created_time DATE,
    following_or_not BOOLEAN,
    like_type TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (photo_id) REFERENCES photos (photo_id)
)
''')
df_likes.to_sql('likes', conn, if_exists='replace', index=False)

# Follows Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS follows (
    follow_id INTEGER PRIMARY KEY,
    follower_user_id INTEGER,
    user_id INTEGER,
    created_time DATE,
    is_follower_active INTEGER,
    followee_acc_status TEXT,
    FOREIGN KEY (follower_user_id) REFERENCES users (user_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
)
''')
df_follows.to_sql('follows', conn, if_exists='replace', index=False)

# Interactions Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    photo_id INTEGER,
    tag_id INTEGER,
    comment_id INTEGER,
    like_id INTEGER,
    follow_id INTEGER,
    interaction_date DATE,
    interaction_type TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (photo_id) REFERENCES photos (photo_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id),
    FOREIGN KEY (comment_id) REFERENCES comments (comment_id),
    FOREIGN KEY (like_id) REFERENCES likes (like_id),
    FOREIGN KEY (follow_id) REFERENCES follows (follow_id)
)
''')
df_interactions.to_sql('interactions', conn, if_exists='replace', index=False)

conn.commit()
conn.close()


# ---------------------------------------------------------
# 4. Target Output Only
# ---------------------------------------------------------

conn = sqlite3.connect(str(db_file))
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM photos")
print(f"Number of photos in photos table: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM users")
print(f"Number of users in users table: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(DISTINCT tag_id) FROM tags")
print(f"Number of tags in tags table: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM interactions WHERE interaction_type = 'like'")
print(f"Number of likes in interactions table: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM interactions WHERE interaction_type = 'comment'")
print(f"Number of comments in interactions table: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM interactions WHERE interaction_type = 'follow'")
print(f"Number of follows in interactions table: {cursor.fetchone()[0]}")

conn.close()
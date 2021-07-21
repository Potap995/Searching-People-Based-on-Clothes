import sqlite3
import sys


def create_db(path="./data/main.db"):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS files")
    cursor.execute("DROP TABLE IF EXISTS tracks_boxes")
    cursor.execute("DROP TABLE IF EXISTS tracks_centers")
    cursor.execute("DROP TABLE IF EXISTS tracks_times")
    cursor.execute("DROP TABLE IF EXISTS tracks_clothes")
    cursor.execute("DROP TABLE IF EXISTS clothes_names")
    cursor.execute("DROP TABLE IF EXISTS tracks_frames")

    cursor.execute("DROP TABLE IF EXISTS info")
    cursor.execute("DROP TABLE IF EXISTS mistakes")
    cursor.execute("DROP TABLE IF EXISTS mistakes_img_info_in")
    cursor.execute("DROP TABLE IF EXISTS mistakes_img_info_out")

    sql_creat_files = """\
    CREATE TABLE files (
        file_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        video_path TEXT NOT NULL UNIQUE,
        processed INTEGER NOT NULL
    );"""

    sql_creat_tracks_boxes = """\
    CREATE TABLE tracks_boxes (
        file_id INTEGER NOT NULL,
        frame_num INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        x_pos REAL NOT NULL, 
        y_pos REAL NOT NULL, 
        width REAL NOT NULL, 
        height REAL NOT NULL,
        FOREIGN KEY (file_id)
        REFERENCES files (file_id) 
           ON UPDATE CASCADE
           ON DELETE CASCADE    
    );"""

    sql_creat_centers = """\
    CREATE TABLE tracks_centers (
        file_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        vector BLOB,
        radius REAL NOT NULL,
        FOREIGN KEY (file_id)
        REFERENCES files (file_id) 
           ON UPDATE CASCADE
           ON DELETE CASCADE
    );"""

    sql_creat_times = """\
    CREATE TABLE tracks_times (
        file_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        start INTEGER NOT NULL,
        end INTEGER NOT NULL,
        FOREIGN KEY (file_id)
        REFERENCES files (file_id) 
           ON UPDATE CASCADE
           ON DELETE CASCADE
    );"""

    sql_creat_clothes = """\
    CREATE TABLE tracks_clothes (
        file_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        clothes INTEGER NOT NULL,
        color TEXT NOT NULL,
        score REAL NOT NULL,
        FOREIGN KEY (file_id)
        REFERENCES files (file_id) 
           ON UPDATE CASCADE
           ON DELETE CASCADE
    );"""

    sql_creat_clothes_names = """\
    CREATE TABLE clothes_names (
        clothes_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );"""

    sql_creat_frames = """\
    CREATE TABLE tracks_frames (
        file_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        average_frame INTEGER NOT NULL,
        frames_num INTEGER NOT NULL,
        FOREIGN KEY (file_id)
        REFERENCES files (file_id) 
           ON UPDATE CASCADE
           ON DELETE CASCADE
        );"""

    sql_creat_index = """\
            CREATE INDEX idx_file_frame 
            ON tracks_boxes (file_id, frame_num);"""

    sql_creat_info = """
    CREATE TABLE info (
        key TEXT NOT NULL,
        value TEXT NOT NULL
    );
    
    INSERT INTO info (key, value)
    VALUES
       ("last_track_id", "0"),
       ("last_save_mistake_id", "0")
    ;
    """

    sql_creat_mistakes = """
    CREATE TABLE mistakes (
        first INT NOT NULL,
        second INT NOT NULL,
        dist REAL NOT NULL,
        type INT NOT NULL
    );
    """

    sql_creat_mistake_img = """
    CREATE TABLE mistakes_img_info_in (
        track_id INT NOT NULL,
        file_id INT NOT NULL,
        frame INT NOT NULL,
        img_id INT NOT NULL
    );
    CREATE TABLE mistakes_img_info_out (
        img_path TEXT NOT NULL,
        img_id INT NOT NULL
    );
    
    CREATE INDEX idx_mistakes_img_out 
    ON mistakes_img_info_out (img_path);
    
    CREATE INDEX idx_mistakes_img_in 
    ON mistakes_img_info_in (track_id);"""

    cursor.execute(sql_creat_files)
    cursor.execute(sql_creat_tracks_boxes)
    cursor.execute(sql_creat_centers)
    cursor.execute(sql_creat_times)
    cursor.execute(sql_creat_clothes)
    cursor.execute(sql_creat_clothes_names)
    cursor.execute(sql_creat_frames) #
    cursor.execute(sql_creat_index)

    cursor.executescript(sql_creat_info)
    cursor.execute(sql_creat_mistakes)
    cursor.executescript(sql_creat_mistake_img)

    conn.commit()

    sql_insert_clothes_names = """\
    INSERT INTO clothes_names 
        (clothes_id, name) 
        VALUES (?,?)
    ;"""

    clothes_f = open("./data/clothes.csv", 'r')
    for line in clothes_f:
        id_, name = line.strip().split(":")
        id_ = int(id_)
        if id_ <= 27:
            cursor.execute(sql_insert_clothes_names, (id_, name))
    clothes_f.close()

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    create_db()

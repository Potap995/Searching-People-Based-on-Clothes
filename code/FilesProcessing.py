from pathlib import Path
import os

class FilesGetter:
    @staticmethod
    def finedAllFiles(video_path):
        parts = list(Path(video_path).parts)
        parts[0] = parts[0].rstrip("\\")
        file_name = parts[-1].rsplit('.', 1)[0]
        file_name_txt = file_name + ".txt"
        track_path = os.path.join("\\".join(parts[:-2]), "tracks", file_name_txt)
        clothes_path = os.path.join("\\".join(parts[:-2]), "clothes", file_name_txt)
        try:
            res = open(track_path, "r")
            res.close()
        except:
            track_path = ""

        try:
            res = open(clothes_path, "r")
            res.close()
        except:
            clothes_path = ""

        ret = [video_path, track_path, clothes_path, file_name]

        return ret


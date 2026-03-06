from dataclasses import dataclass, field


@dataclass
class DownloadConfig:
    subjects: list[int]
    runs: list[int]
    data_path: str

import os


class Config:
    model: str = os.getenv("MODEL", "gemini-3.1-flash-lite")
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "100.0"))


config = Config()

"""Reema: persistent, short process notes for learning across runs."""
import json
from pathlib import Path


class AgentMemory:
    """Keep recent research-process lessons per ticker in a local JSON file."""

    def __init__(self, path="research_memory.json", max_notes=3):
        self.path = Path(path)
        self.max_notes = max_notes

    def _load(self):
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as file:
            return json.load(file)

    def get(self, ticker):
        """Retrieve prior notes for this ticker."""
        return self._load().get(ticker.upper(), [])[-self.max_notes:]

    def save(self, ticker, report):
        """Save compact notes, not full price histories or news datasets."""
        data = self._load()
        key = ticker.upper()
        data.setdefault(key, []).append({
            "question": report["question"],
            "routes": report["plan"]["routes"],
            "reflection": report["reflection"],
        })
        data[key] = data[key][-self.max_notes:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

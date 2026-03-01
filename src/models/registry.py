from pathlib import Path
import datetime
import joblib
from typing import Any, Optional

class ModelRegistry:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True,exist_ok=True)

    def _timestamp(self):
        return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save_model(self, model:Any, name:str, with_timestamp=True)->Path:
        if with_timestamp:
            stem=Path(name).stem
            suffix=Path(name).suffix or ".pkl"
            filename=f"{stem}_{self._timestamp()}{suffix}"
        else: filename=name
        path=self.model_dir/filename
        joblib.dump(model,path)
        return path

    def load_model(self,path:Optional[str]=None,latest=False)->Any:
        if latest:
            candidates=sorted(self.model_dir.glob("*.pkl"))
            if not candidates: raise FileNotFoundError("No models found")
            path=candidates[-1]
        elif path is not None:
            path=Path(path)
        else: raise ValueError("Provide path or latest=True")
        return joblib.load(path)
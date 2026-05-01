import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetBundle:
    queries: pd.DataFrame
    cases: pd.DataFrame
    statutes: pd.DataFrame
    case_qrels: pd.DataFrame
    statute_qrels: pd.DataFrame


class AilaDataLoader:
    def __init__(self, root_dir: str | Path = "."):
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.query_dir = self.data_dir / "queries"
        self.case_dir = self.data_dir / "cases"
        self.statute_dir = self.data_dir / "statutes"
        self.qrels_dir = self.data_dir / "qrels"
        self.processed_dir = self.data_dir / "processed"

    def download_dataset(self) -> Path:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub is not installed. Run: pip install -r requirements.txt"
            ) from exc

        path = Path(kagglehub.dataset_download("ananyapam7/legalai")).resolve()
        return path

    def prepare_dataset(self, source_path: str | Path | None = None, force: bool = False) -> None:
        for directory in [
            self.raw_dir,
            self.query_dir,
            self.case_dir,
            self.statute_dir,
            self.qrels_dir,
            self.processed_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        if source_path is not None:
            self._copy_or_extract(Path(source_path), self.raw_dir, force=force)

        query_file = self._find_one("Query_doc.txt")
        case_source = self._find_dir("Object_casedocs")
        statute_source = self._find_dir("Object_statutes")

        if query_file is None:
            raise FileNotFoundError(
                "Could not find Query_doc.txt. Check the Kaggle download or place it under data/raw."
            )
        if case_source is None:
            raise FileNotFoundError(
                "Could not find Object_casedocs. Check the Kaggle download or place case files under data/raw."
            )
        if statute_source is None:
            raise FileNotFoundError(
                "Could not find Object_statutes. Check the Kaggle download or place statute files under data/raw."
            )

        shutil.copy2(query_file, self.query_dir / "Query_doc.txt")
        self._copy_txt_files(case_source, self.case_dir, prefix="C", force=force)
        self._copy_txt_files(statute_source, self.statute_dir, prefix="S", force=force)

        for name in [
            "relevance_judgments_priorcases.txt",
            "relevance_judgments_statutes.txt",
        ]:
            qrel_file = self._find_one(name)
            if qrel_file is not None:
                shutil.copy2(qrel_file, self.qrels_dir / name)

    def parse_all(self) -> DatasetBundle:
        queries = self.parse_queries(self.query_dir / "Query_doc.txt")
        cases = self.parse_cases(self.case_dir)
        statutes = self.parse_statutes(self.statute_dir)
        case_qrels = self.parse_qrels(
            self.qrels_dir / "relevance_judgments_priorcases.txt"
        )
        statute_qrels = self.parse_qrels(
            self.qrels_dir / "relevance_judgments_statutes.txt"
        )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        queries.to_csv(self.processed_dir / "queries.csv", index=False)
        cases.to_csv(self.processed_dir / "cases.csv", index=False)
        statutes.to_csv(self.processed_dir / "statutes.csv", index=False)
        case_qrels.to_csv(self.processed_dir / "case_qrels.csv", index=False)
        statute_qrels.to_csv(self.processed_dir / "statute_qrels.csv", index=False)

        for name, frame in {
            "queries": queries,
            "cases": cases,
            "statutes": statutes,
            "case_qrels": case_qrels,
            "statute_qrels": statute_qrels,
        }.items():
            frame.to_json(
                self.processed_dir / f"{name}.json",
                orient="records",
                indent=2,
                force_ascii=False,
            )

        metadata = {
            "num_queries": int(len(queries)),
            "num_cases": int(len(cases)),
            "num_statutes": int(len(statutes)),
            "num_case_qrels": int(len(case_qrels)),
            "num_statute_qrels": int(len(statute_qrels)),
        }
        (self.processed_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return DatasetBundle(queries, cases, statutes, case_qrels, statute_qrels)

    def load_processed(self) -> DatasetBundle:
        required = [
            "queries.csv",
            "cases.csv",
            "statutes.csv",
            "case_qrels.csv",
            "statute_qrels.csv",
        ]
        missing = [name for name in required if not (self.processed_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Processed files are missing: {missing}. Run python main.py first."
            )
        return DatasetBundle(
            pd.read_csv(self.processed_dir / "queries.csv").fillna(""),
            pd.read_csv(self.processed_dir / "cases.csv").fillna(""),
            pd.read_csv(self.processed_dir / "statutes.csv").fillna(""),
            pd.read_csv(self.processed_dir / "case_qrels.csv").fillna(""),
            pd.read_csv(self.processed_dir / "statute_qrels.csv").fillna(""),
        )

    @staticmethod
    def parse_queries(path: str | Path) -> pd.DataFrame:
        records = []
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or "||" not in line:
                continue
            query_id, text = line.split("||", 1)
            records.append({"query_id": query_id.strip(), "text": text.strip()})
        if not records:
            raise ValueError(f"No valid queries found in {path}")
        return pd.DataFrame(records).drop_duplicates("query_id")

    @staticmethod
    def parse_cases(directory: str | Path) -> pd.DataFrame:
        records = []
        for file_path in sorted(Path(directory).glob("C*.txt"), key=AilaDataLoader._natural_key):
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            records.append({"doc_id": file_path.stem, "text": text, "path": str(file_path)})
        if not records:
            raise ValueError(f"No case documents found in {directory}")
        return pd.DataFrame(records).drop_duplicates("doc_id")

    @staticmethod
    def parse_statutes(directory: str | Path) -> pd.DataFrame:
        records = []
        for file_path in sorted(Path(directory).glob("S*.txt"), key=AilaDataLoader._natural_key):
            raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            title = ""
            desc_parts = []
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("title:"):
                    title = stripped.split(":", 1)[1].strip()
                elif stripped.lower().startswith("desc:"):
                    desc_parts.append(stripped.split(":", 1)[1].strip())
                elif stripped:
                    desc_parts.append(stripped)
            description = " ".join(desc_parts).strip()
            text = f"{title}. {description}".strip(". ")
            records.append(
                {
                    "doc_id": file_path.stem,
                    "title": title,
                    "description": description,
                    "text": text,
                    "path": str(file_path),
                }
            )
        if not records:
            raise ValueError(f"No statute documents found in {directory}")
        return pd.DataFrame(records).drop_duplicates("doc_id")

    @staticmethod
    def parse_qrels(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        columns = ["query_id", "iter", "doc_id", "relevance"]
        if not path.exists():
            return pd.DataFrame(columns=columns)

        records = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                relevance = int(float(parts[3]))
            except ValueError:
                relevance = 0
            records.append(
                {
                    "query_id": parts[0],
                    "iter": parts[1],
                    "doc_id": parts[2],
                    "relevance": relevance,
                }
            )
        return pd.DataFrame(records, columns=columns)

    def _copy_or_extract(self, source: Path, destination: Path, force: bool = False) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source}")

        if source.is_file() and source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                zf.extractall(destination)
            return

        if source.is_dir():
            for child in source.iterdir():
                target = destination / child.name
                if target.exists() and not force:
                    continue
                if child.is_dir():
                    if target.exists() and force:
                        shutil.rmtree(target)
                    shutil.copytree(child, target)
                elif child.suffix.lower() == ".zip":
                    with zipfile.ZipFile(child) as zf:
                        zf.extractall(destination / child.stem)
                else:
                    shutil.copy2(child, target)
            return

        shutil.copy2(source, destination / source.name)

    def _find_one(self, filename: str) -> Path | None:
        candidates = list(self.raw_dir.rglob(filename)) + list(self.data_dir.rglob(filename))
        candidates = [path for path in candidates if self.processed_dir not in path.parents]
        return candidates[0] if candidates else None

    def _find_dir(self, dirname: str) -> Path | None:
        candidates = [path for path in self.raw_dir.rglob(dirname) if path.is_dir()]
        candidates += [path for path in self.data_dir.rglob(dirname) if path.is_dir()]
        if candidates:
            return candidates[0]

        prefix = "C" if "case" in dirname.lower() else "S"
        fallback = [
            path
            for path in [self.raw_dir, self.data_dir]
            if path.exists() and any(path.glob(f"{prefix}*.txt"))
        ]
        return fallback[0] if fallback else None

    @staticmethod
    def _copy_txt_files(source: Path, destination: Path, prefix: str, force: bool = False) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for file_path in source.glob(f"{prefix}*.txt"):
            target = destination / file_path.name
            if target.exists() and not force:
                continue
            shutil.copy2(file_path, target)

    @staticmethod
    def _natural_key(path: Path):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else stem

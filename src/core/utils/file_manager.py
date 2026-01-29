# core/utils/file_manager.py

# --- Standard Python imports ---
from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, List, Iterable
import zipfile

# --- Third-party imports ---
import joblib
import pandas as pd
import yaml

# --- Local imports ---
from core.utils.io import ensure_directories
from core.utils.logger import get_logger

# base logger (no handlers; propagates to root)
log = get_logger(__name__) 

# ---------------------
# | Base File Manager |
# ---------------------

class FileManagerStrategy(ABC):
    """
    Abstract base class to represent a file management strategy.

    Attributes:
        log (logging.Logger): Logger instance for logging events.
    """
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes the FileManagerStrategy with an optional logger.

        Args:
            logger (Optional[logging.Logger]): A logger to be used by the strategy (default None).
        """
        base = logger or log
        self.log =  base.getChild(self.__class__.__name__)

    @abstractmethod
    def files(
        self,
        file_content: Optional[Any] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        file_format: Optional[Any] = None,
        **kwargs
    ) -> Any: ...
    """
    Handling the content of a file.

    Args:
        file_content (Optional[Any]): A particular file content.
        input_path (Optional[str]): The path to the file to read.
        output_path (Optional[str]): The path to the file to write.

        file_format (Optional[Any]): 
            The specific extension of a file (e.g., ".csv", ".joblib" etc.)
    
    Returns:
        Any: The processed result, which can also be of any type.
    """

    @staticmethod
    def get_ext(path: Optional[str]) -> Optional[str]:
        """
        Get extension split path to find the file extensions.
        """
        return os.path.splitext(path)[1].lower() if path else None
    
    @staticmethod
    def norm_ext(e: str) -> str:
        """
        Normalize extension ensure "." before the file format/ extensions.
        """
        e = e.strip().lower()
        return e if e.startswith(".") else "." + e

# ---------------
# | Zip Manager |
# ---------------
class ZipManager(FileManagerStrategy):
    """
    Concrete implementation of the FileManagerStrategy for handling Zipped files.
    """
    def files(
        self,
        file_content: Optional[Any] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        file_format: Optional[Any] = None,
        on_conflict: str = "overwrite",  # 
        **kwargs
    ) -> List[str]:
        
        """
        To read and write the Zipped files.

        Can handle similar or multiple file extensions to read from ZIP and write in the output path.

        Args:
            file_content (Optional[Any]): A particular file content.
            input_path (Optional[str]): The path to the file to read the ZIP.
            output_path (Optional[str]): The path to the files in the ZIP to write.

            file_format (Optional[Any]): 
                Specific extensions file to be select from ZIP (e.g., ".csv", ".joblib", [".txt", ".yaml"] etc.)

            on_conflict (str): 
                Handling duplicate file in the output path ["overwrite" or "skip" or "rename"] (default "overwrite")

        Returns:
            List[str]: A list of strings with ZIP extreacted files.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if input_path is None or output_path is None:
            raise ValueError("Provide `input_path` (.zip) and `output_path` (path).")
        
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if self.get_ext(input_path) != ".zip":
            raise FileNotFoundError(f"Zip file not found: {input_path}")

        self.log.info("extract.begin input=%s out=%s", input_path, output_path)

        # Ensure the output directory exits or not
        ensure_directories(output_path)
        # Resolve to absolute path
        base = Path(output_path).resolve()

        if file_format is None:
            exts = None
        elif isinstance(file_format, str):
            exts = {self.norm_ext(file_format)} 
        elif isinstance(file_format, Iterable):
            exts = {self.norm_ext(x) for x in file_format} or None
        else:
            raise TypeError("`file_format` must be None, str, or iterable[str].")

        extracted = []

        def get_target_path(member: str) -> Path:
            name = os.path.basename(member)  # flatten
            t = (base / name).resolve()
            if not str(t).startswith(str(base)):  # zip-slip guard
                raise ValueError(f"Unsafe path in zip: {member}")
            return t

        with zipfile.ZipFile(input_path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            
            if exts:
                members = [m for m in members if any(m.lower().endswith(e) for e in exts)]
            self.log.debug("members.to_extract= %d filter=%s", len(members), list(exts) if exts else None)

            for m in members:
                t = get_target_path(m)
                if t.exists():
                    if on_conflict == "skip":
                        self.log.debug("skip.exists %s", t)
                        continue
                    if on_conflict == "rename":
                        stem, suf, i = t.stem, t.suffix, 1
                        while t.exists():
                            t = base / f"{stem}({i}){suf}"
                            i += 1
                with zf.open(m) as src, open(t, "wb") as dst:
                    dst.write(src.read())
                extracted.append(str(t))

                if self.log.isEnabledFor(logging.DEBUG):
                    self.log.debug("extracted %s", t)

        self.log.info("extract.success files=%d out=%s", len(extracted), output_path)
        
        return extracted

# ----------------
# | Read Manager |
# ----------------
class ReadManager(FileManagerStrategy):
    """
    Concrete implementation of the FileManagerStrategy for reading files.
    """
    def files(
        self,
        file_content: Optional[Any] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        file_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        
        """
        To read the content in a file.

        Args:
            file_content (Optional[Any]): A particular file content.
            input_path (Optional[str]): The path to the file to read.
            output_path (Optional[str]): The path to the file to write.

            file_format (Optional[Any]): 
                The specific extension of a file (e.g., ".csv", ".joblib" etc.)

        Returns:
            Any: The processed result, which can also be of any type.
        """
        input_path = Path(input_path)

        if input_path is None:
            raise ValueError("Provide `input_path` to read.")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ext = self.get_ext(input_path) # get file extension
        if not ext:
            raise ValueError(f"Cannot infer file extension: {input_path}")

        self.log.info("read.begin path=%s ext=%s", input_path, ext)

        if ext == ".csv":
            content = pd.read_csv(input_path)

        elif ext == ".tsv":
            content = pd.read_csv(input_path, sep="\t")

        elif ext == ".txt":
            with open(input_path, 'r') as file:
                obj = file.readlines()
            self.log.info("read.success type=txt")
            return obj

        elif ext in [".xlsx", ".xls"]:
            content = pd.read_excel(input_path)

        elif ext in [".yaml", ".yml"]:
            with open(input_path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
            self.log.info("read.success type=yaml")
            return obj 
        
        elif ext == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.log.info("read.success type=json")
            return obj 
        
        elif ext == ".parquet":
            try:
                content = pd.read_parquet(input_path)
            except ImportError as e:
                self.log.error("parquet.engine.missing: %s", e)
                raise
            
        elif ext == ".joblib":
            obj = joblib.load(input_path)
            self.log.info("read.success type=joblib")
            return obj
        else:
            raise ValueError(f"Unsupported read extension: {ext}")

        if isinstance(content, pd.DataFrame):
            self.log.info("read.success type=df shape=%s", getattr(content, "shape", None))

        return content

# -----------------
# | Write Manager |
# -----------------
class WriteManager(FileManagerStrategy):
    """
    Concrete implementation of the FileManagerStrategy for writing files.
    """
    def files(
        self,
        file_content: Optional[Any] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        file_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        
        """
        To write the content in a file.

        Args:
            file_content (Optional[Any]): A particular file content.
            input_path (Optional[str]): The path to the file to read.
            output_path (Optional[str]): The path to the file to write.

            file_format (Optional[Any]): 
                The specific extension of a file (e.g., ".csv", ".joblib" etc.)

        Returns:
            str: A success message.
        """
        output_path = Path(output_path)
                
        if file_content is None or output_path is None:
            raise ValueError("Provide `file_content` and `output_path` to write.")

        # Ensure the output directory exits or not
        ensure_directories(Path(output_path).parent)

        ext = self.get_ext(output_path) # get extension
        if not ext:
            raise ValueError(f"Cannot infer file extension: {output_path}")

        self.log.info("write.begin path=%s ext=%s", output_path, ext)

        if ext == ".csv":
            assert isinstance(file_content, pd.DataFrame)
            file_content.to_csv(output_path, index=False)

        elif ext == ".tsv":
            assert isinstance(file_content, pd.DataFrame)
            file_content.to_csv(output_path, sep="\t", index=False)

        elif ext == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(file_content)

        elif ext in [".xlsx", ".xls"]:
            assert isinstance(file_content, pd.DataFrame)
            file_content.to_excel(output_path, index=False)

        elif ext == ".parquet":
            assert isinstance(file_content, pd.DataFrame)
            try:
                file_content.to_parquet(output_path, index=False)
            except ImportError as e:
                self.log.error("parquet.engine.missing: %s", e)
                raise

        elif ext in [".yaml", ".yml"]:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(file_content, f, sort_keys=False, allow_unicode=True)

        elif ext == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(file_content, f, indent=4, ensure_ascii=False)

        elif ext == ".joblib":
            joblib.dump(file_content, output_path, compress= 3)
            
        else:
            raise ValueError(f"Unsupported write extension: {ext}")

        if isinstance(file_content, pd.DataFrame):
            return self.log.info("write.success type=df shape=%s", file_content.shape)
        else:
            return self.log.info("write.success type=%s", type(file_content).__name__)

"""
Native Checkpointer & State Persistence Engine for tech-rkj.

Implements the .rkj v2.0 binary container format with:
- 64-byte aligned header with magic bytes b"RKJ\x02"
- CRC32 and SHA-256 payload integrity checksums
- JSON metadata block (model architecture, hyperparams, schema, timestamp)
- Memory-mapped zero-copy deserialization (mmap)
- Atomic write-rename-replace rollback safety
"""

import os
import sys
import json
import zlib
import hashlib
import mmap
import time
from pathlib import Path
from typing import Union, Optional, Any, Dict

import numpy as np

MAGIC_V2 = b"RKJ\x02"
HEADER_SIZE = 64

MODEL_TYPE_REGRESSOR = 0x01
MODEL_TYPE_CLASSIFIER = 0x02

CHECKSUM_CRC32 = 0x01
CHECKSUM_SHA256 = 0x02


def save(
    model: Any,
    path: Union[str, Path],
    checksum: str = "crc32",
) -> str:
    """
    Save a fitted tech-rkj model to an .rkj binary container file with atomic
    write-replace rollback safety and payload integrity checksum.

    Parameters
    ----------
    model : SubspaceRegressor or SubspaceClassifier
        Fitted model instance.
    path : str or Path
        Target destination file path.
    checksum : str, default="crc32"
        Checksum algorithm: "crc32" (fastest) or "sha256" (cryptographic).

    Returns
    -------
    str : Absolute path to the saved checkpoint file.
    """
    if not getattr(model, "is_fitted_", False) or getattr(model, "_core", None) is None:
        raise ValueError("Cannot save an unfitted model. Call fit() first.")

    target_path = Path(path).resolve()
    checksum_lower = checksum.lower().strip()
    if checksum_lower == "crc32":
        chk_type = CHECKSUM_CRC32
    elif checksum_lower == "sha256":
        chk_type = CHECKSUM_SHA256
    else:
        raise ValueError(
            f"Unsupported checksum '{checksum}'. Supported algorithms: 'crc32', 'sha256'."
        )

    # Determine model type
    from tech_rkj.classifier import SubspaceClassifier
    from tech_rkj.regressor import SubspaceRegressor

    if isinstance(model, SubspaceRegressor):
        model_type = MODEL_TYPE_REGRESSOR
        type_str = "SubspaceRegressor"
    elif isinstance(model, SubspaceClassifier):
        model_type = MODEL_TYPE_CLASSIFIER
        type_str = "SubspaceClassifier"
    else:
        raise ValueError(f"Unsupported model type for .rkj format: {type(model)}")

    # Extract binary payload from native Rust core
    core_bytes = bytes(model._core.__getstate__())
    payload_len = len(core_bytes)

    # Compute checksum
    if chk_type == CHECKSUM_CRC32:
        crc_val = zlib.crc32(core_bytes)
        chk_bytes = crc_val.to_bytes(4, "little") + (b"\x00" * 28)
    else:
        chk_bytes = hashlib.sha256(core_bytes).digest()

    # Build metadata dictionary
    meta_dict: Dict[str, Any] = {
        "format": "rkj",
        "format_version": "2.0",
        "package_version": "0.2.0",
        "created_at": time.time(),
        "platform": sys.platform,
        "model_type": type_str,
        "num_heads": model.num_heads,
        "head_circuits": model.head_circuits,
        "alpha": model.alpha,
        "forgetting_factor": model.forgetting_factor,
        "low_freq_scale": model.low_freq_scale,
        "high_freq_scale": model.high_freq_scale,
        "frontend_mode": model.frontend_mode,
        "n_features_in_": model.n_features_in_,
        "n_samples_seen_": model.n_samples_seen_,
    }

    if model_type == MODEL_TYPE_CLASSIFIER:
        classes = model.classes_
        meta_dict["classes_"] = (
            classes.tolist() if hasattr(classes, "tolist") else list(classes)
        )
        meta_dict["n_classes_"] = model.n_classes_

    meta_json = json.dumps(meta_dict, separators=(",", ":")).encode("utf-8")
    meta_len = len(meta_json)

    # Construct 64-byte aligned binary header
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC_V2
    header[4] = model_type
    header[5] = chk_type
    header[6:8] = b"\x00\x00"  # flags / reserved
    header[8:12] = meta_len.to_bytes(4, "little")
    header[12:20] = payload_len.to_bytes(8, "little")
    header[20:52] = chk_bytes
    header[52:64] = b"\x00" * 12  # padding

    # Atomic write-rename-replace
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(
        f".{target_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )

    try:
        with open(temp_path, "wb") as f:
            f.write(header)
            f.write(meta_json)
            f.write(core_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return str(target_path)


def load(
    path: Union[str, Path],
    expected_type: Optional[int] = None,
    use_mmap: bool = True,
) -> Any:
    """
    Load a model from an .rkj binary container file with integrity verification.

    Parameters
    ----------
    path : str or Path
        Path to the .rkj file.
    expected_type : int, optional
        MODEL_TYPE_REGRESSOR (0x01) or MODEL_TYPE_CLASSIFIER (0x02) to enforce type validation.
    use_mmap : bool, default=True
        Whether to memory-map the file for fast zero-copy deserialization.

    Returns
    -------
    SubspaceRegressor or SubspaceClassifier
        Reconstructed, fully fitted model instance.
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file '{path_obj}' does not exist.")

    file_size = path_obj.stat().st_size
    if file_size < HEADER_SIZE:
        raise ValueError(
            f"Corrupted or invalid checkpoint: file size {file_size} bytes is smaller than header ({HEADER_SIZE} bytes)."
        )

    with open(path_obj, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("Unexpected EOF reading checkpoint header.")

        # 1. Magic byte verification
        magic = bytes(header[0:4])
        if magic != MAGIC_V2:
            raise ValueError(
                f"Invalid magic header: {magic!r}. Expected {MAGIC_V2!r} (not an .rkj v2 container)."
            )

        model_type = header[4]
        chk_type = header[5]
        meta_len = int.from_bytes(header[8:12], "little")
        payload_len = int.from_bytes(header[12:20], "little")
        stored_chk = bytes(header[20:52])

        # 2. Model type validation
        if expected_type is not None and model_type != expected_type:
            expected_name = (
                "SubspaceRegressor"
                if expected_type == MODEL_TYPE_REGRESSOR
                else "SubspaceClassifier"
            )
            found_name = (
                "SubspaceClassifier"
                if model_type == MODEL_TYPE_CLASSIFIER
                else "SubspaceRegressor"
            )
            raise ValueError(
                f"Model type mismatch: checkpoint contains {found_name}, expected {expected_name}."
            )

        # 3. File length consistency
        expected_total = HEADER_SIZE + meta_len + payload_len
        if file_size < expected_total:
            raise ValueError(
                f"Corrupted checkpoint: file truncated. Expected {expected_total} bytes, got {file_size} bytes."
            )

        # 4. Read metadata & payload
        if use_mmap:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                meta_bytes = bytes(mm[HEADER_SIZE : HEADER_SIZE + meta_len])
                payload_bytes = bytes(
                    mm[HEADER_SIZE + meta_len : HEADER_SIZE + meta_len + payload_len]
                )
        else:
            f.seek(HEADER_SIZE)
            meta_bytes = f.read(meta_len)
            payload_bytes = f.read(payload_len)

    # 5. Checksum verification
    if chk_type == CHECKSUM_CRC32:
        expected_crc = int.from_bytes(stored_chk[:4], "little")
        actual_crc = zlib.crc32(payload_bytes)
        if actual_crc != expected_crc:
            raise ValueError(
                f"CRC32 integrity check failed: expected {hex(expected_crc)}, computed {hex(actual_crc)}. File is corrupted."
            )
    elif chk_type == CHECKSUM_SHA256:
        actual_sha = hashlib.sha256(payload_bytes).digest()
        if actual_sha != stored_chk:
            raise ValueError(
                "SHA-256 integrity check failed: payload hash mismatch. File is corrupted."
            )
    else:
        raise ValueError(f"Unknown checksum algorithm flag: {chk_type}")

    # 6. Parse metadata
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON metadata in checkpoint: {e}") from e

    # 7. Model reconstruction
    from tech_rkj.classifier import SubspaceClassifier
    from tech_rkj.regressor import SubspaceRegressor

    if model_type == MODEL_TYPE_REGRESSOR:
        model = SubspaceRegressor(
            num_heads=meta["num_heads"],
            head_circuits=meta["head_circuits"],
            alpha=meta["alpha"],
            forgetting_factor=meta["forgetting_factor"],
            low_freq_scale=meta["low_freq_scale"],
            high_freq_scale=meta["high_freq_scale"],
            frontend_mode=meta["frontend_mode"],
        )
        model.n_features_in_ = meta["n_features_in_"]
        model.n_samples_seen_ = meta["n_samples_seen_"]
        model.is_fitted_ = True
        model._init_core(model.n_features_in_)
        try:
            model._core.__setstate__(payload_bytes)
        except Exception as e:
            raise ValueError(f"Native deserialization error: {e}") from e
        return model

    elif model_type == MODEL_TYPE_CLASSIFIER:
        model = SubspaceClassifier(
            num_heads=meta["num_heads"],
            head_circuits=meta["head_circuits"],
            alpha=meta["alpha"],
            forgetting_factor=meta["forgetting_factor"],
            low_freq_scale=meta["low_freq_scale"],
            high_freq_scale=meta["high_freq_scale"],
            frontend_mode=meta["frontend_mode"],
        )
        model.classes_ = np.array(meta["classes_"])
        model.n_classes_ = meta["n_classes_"]
        model.n_features_in_ = meta["n_features_in_"]
        model.n_samples_seen_ = meta["n_samples_seen_"]
        model.is_fitted_ = True
        model._init_core(model.n_features_in_, model.n_classes_)
        try:
            model._core.__setstate__(payload_bytes)
        except Exception as e:
            raise ValueError(f"Native deserialization error: {e}") from e
        return model

    else:
        raise ValueError(f"Unrecognized model type tag: {model_type}")

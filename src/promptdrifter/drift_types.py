import os
import re
from typing import Pattern


def exact_match(expected_output: str, actual_output: str) -> bool:
    return expected_output == actual_output


def regex_match(regex_pattern: str | Pattern[str], actual_output: str) -> bool:
    try:
        if isinstance(regex_pattern, re.Pattern):
            return bool(regex_pattern.search(actual_output))
        else:
            compiled_regex = re.compile(regex_pattern)
            return bool(compiled_regex.search(actual_output))
    except re.error:
        return False


def expect_substring(substring: str, actual_output: str) -> bool:
    return substring in actual_output


def expect_substring_case_insensitive(substring: str, actual_output: str) -> bool:
    return substring.lower() in actual_output.lower()

def text_similarity(expected_output: str, actual_output: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        raise ImportError(
            "text_similarity requires the 'sentence-transformers' package. "
            "Install it with: pip install 'promptdrifter[similarity]' "
            "or: pip install sentence-transformers"
        )

    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        emb = model.encode([expected_output, actual_output], convert_to_tensor=True)
        score = util.cos_sim(*emb).item()
        return score
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

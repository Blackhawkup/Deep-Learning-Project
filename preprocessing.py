import re
import unicodedata
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\x00", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s.,;:()'/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    if remove_stopwords:
        tokens = [tok for tok in text.split() if tok not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)

    return text


def clean_legal_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    strip_citations: bool = True,
    strip_case_numbers: bool = True,
    normalize_statutes: bool = True,
) -> str:
    """Enhanced text cleaning for legal documents.

    Removes noisy patterns common in Indian Supreme Court judgments that
    confuse retrieval models without adding semantic content:
    - Case citation patterns (e.g., "AIR 1995 SC 123")
    - Case numbers (e.g., "Civil Appeal No. 1234 of 2005")
    - SCC/AIR volume references
    - Dates in various formats
    - Judge names and bench composition lines
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\x00", " ")

    if strip_citations:
        # AIR citations: "AIR 1995 SC 123", "(1995) 3 SCC 456"
        text = re.sub(r"\bAIR\s+\d{4}\s+\w+\s+\d+", " ", text)
        text = re.sub(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+", " ", text)
        text = re.sub(r"\(\d{4}\)\s*\d+\s*SCC\s*\(\w+\)\s*\d+", " ", text)
        text = re.sub(r"\[\d{4}\]\s*\d+\s*SCR\s*\d+", " ", text)
        # General citation patterns
        text = re.sub(r"\d+\s+SCC\s+\d+", " ", text)
        text = re.sub(r"\d+\s+SCR\s+\d+", " ", text)

    if strip_case_numbers:
        # "Civil Appeal No. 1234 of 2005"
        text = re.sub(
            r"(Civil|Criminal|Writ|Special Leave)\s*(Appeal|Petition|Application)"
            r"\s*No\.?\s*\d+[\s/-]*\d*\s*(of\s+\d{4})?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        # "SLP (C) No. 12345 of 2010"
        text = re.sub(r"SLP\s*\([A-Z]+\)\s*No\.?\s*\d+", " ", text)
        # "W.P. (C) No. 1234/2015"
        text = re.sub(r"[A-Z]\.\s*[A-Z]\.\s*\([A-Z]+\)\s*No\.?\s*\d+", " ", text)

    if normalize_statutes:
        # Normalize "Section 302 of IPC" -> "section 302 ipc"
        text = re.sub(
            r"[Ss]ection\s+(\d+[A-Za-z]*)\s+of\s+(the\s+)?",
            r"section \1 ",
            text,
        )

    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove dates (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy)
    text = re.sub(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", " ", text)

    # Remove standalone numbers (case numbers, page numbers etc.)
    text = re.sub(r"\b\d{3,}\b", " ", text)  # 3+ digit numbers

    # Clean special characters but preserve meaningful punctuation
    text = re.sub(r"[^A-Za-z0-9\s.,;:()'/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    if remove_stopwords:
        tokens = [tok for tok in text.split() if tok not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)

    return text


def extract_legal_sections(text: str) -> dict[str, str]:
    """Attempt to extract structured sections from a legal judgment.

    Returns a dict with keys like 'facts', 'issues', 'arguments',
    'analysis', 'conclusion'. Falls back to splitting by position if
    no explicit section headers are found.
    """
    sections = {
        "full": text,
        "first_quarter": "",
        "middle_half": "",
        "last_quarter": "",
    }

    words = text.split()
    n = len(words)
    if n == 0:
        return sections

    q1 = n // 4
    q3 = 3 * n // 4
    sections["first_quarter"] = " ".join(words[:q1])
    sections["middle_half"] = " ".join(words[q1:q3])
    sections["last_quarter"] = " ".join(words[q3:])

    # Try to find explicit section markers in Indian judgments
    lower = text.lower()
    for marker, key in [
        ("the facts of the case", "facts"),
        ("the facts briefly", "facts"),
        ("brief facts", "facts"),
        ("the questions? (?:of law |for consideration |that arise)", "issues"),
        ("issues? (?:for|that|which)", "issues"),
        ("contentions? (?:of|raised|advanced)", "arguments"),
        ("submissions? (?:of|made|advanced)", "arguments"),
        ("(?:we|i) (?:have considered|now proceed|shall now)", "analysis"),
        ("in (?:our|my) (?:opinion|view|considered)", "analysis"),
        ("for (?:the|these) reasons", "conclusion"),
        ("(?:the appeal|the petition) is (?:accordingly|hereby|therefore)", "conclusion"),
        ("order", "conclusion"),
    ]:
        match = re.search(marker, lower)
        if match:
            start = match.start()
            sections[key] = text[start : start + len(text) // 3]

    return sections


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    text = clean_text(text, lowercase=True, remove_stopwords=False)
    tokens = re.findall(r"[a-z0-9]+", text)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS]
    return tokens


def preprocess_corpus(
    texts: Iterable[str],
    lowercase: bool = True,
    remove_stopwords: bool = False,
    legal_cleaning: bool = False,
) -> list[str]:
    clean_fn = clean_legal_text if legal_cleaning else clean_text
    return [
        clean_fn(text, lowercase=lowercase, remove_stopwords=remove_stopwords)
        for text in texts
    ]

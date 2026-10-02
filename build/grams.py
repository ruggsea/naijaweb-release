"""The 13-word sequences used for decontamination (make_bins.py) and its checks (wiki_decontam.py)."""
import hashlib, re, unicodedata
NGRAM = 13
WORD = re.compile(r"[\ẁ-ͯ]+")

def grams(text):
    w = WORD.findall(unicodedata.normalize("NFC", text).lower())
    return {hashlib.blake2b(" ".join(w[i:i + NGRAM]).encode(), digest_size=8).digest()
            for i in range(len(w) - NGRAM + 1)}


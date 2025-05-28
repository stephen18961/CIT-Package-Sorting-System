import csv
import random
import re
from heapq import nlargest
from rapidfuzz import fuzz

random.seed(42)

def extract_ocr_words(ocr_text):
    """
    Extracts just the words from OCR text that includes coordinates.
    Each line in OCR text is formatted as: "word, (x1, y1), ..."
    """
    words = []
    for line in ocr_text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(',', 1)
        if parts and parts[0].strip():
            words.append(parts[0].strip().lower())
    return words


def remove_noise_words(words, noise_words, threshold=0.8):
    """
    Removes tokens that approximately match any noise word.
    """
    cleaned = []
    for w in words:
        if not any(fuzz.ratio(w, n) / 100.0 > threshold for n in noise_words):
            cleaned.append(w)
    return cleaned


def fuzzy_token_set_ratio(name, words):
    tokens = name.lower().split()
    if not tokens:
        return 0
    total = 0
    for t in tokens:
        best = max((fuzz.ratio(t, w) / 100.0 for w in words), default=0)
        total += best
    return total / len(tokens)


def full_name_concatenation_score(candidate, words):
    """
    Checks fuzzy-similarity of full name (no spaces) against each OCR token.
    """
    name = candidate[0].lower().replace(" ", "")
    if len(name) < 6:
        return 0
    best = 0
    for w in words:
        if w == name:
            return 1.0
        score = fuzz.ratio(w, name) / 100.0
        if score > best:
            best = score
    return best


def score_candidate(candidate, words, threshold=0.8):
    name = candidate[0]
    tokens = name.lower().split()
    if not tokens:
        return 0

    # Full concatenation check
    full_score = full_name_concatenation_score(candidate, words)

    # First & last token matching
    first, last = tokens[0], tokens[-1]
    first_r = max((fuzz.ratio(first, w)/100.0 for w in words), default=0)
    last_r  = max((fuzz.ratio(last,  w)/100.0 for w in words), default=0)

    if first_r > threshold and last_r > threshold:
        base = 1.0
    elif first_r > threshold or last_r > threshold:
        base = 0.7
    else:
        base = 0.0
    # bonus for middle tokens
    bonus = 0
    if len(tokens) > 2:
        mids = tokens[1:-1]
        ratios = [max((fuzz.ratio(m, w)/100.0 for w in words), default=0) for m in mids]
        bonus = sum(r for r in ratios if r > threshold) / len(mids) * 0.3
    first_last = min(1.0, base + bonus)

    # Token set ratio
    set_ratio = fuzzy_token_set_ratio(name, words)

    # All-parts match
    matched = sum(1 for t in tokens if max((fuzz.ratio(t, w)/100.0 for w in words), default=0) > threshold)
    all_parts = matched / len(tokens)

    # Token frequency
    freq = sum(1 for t in tokens for w in words if t == w)
    token_freq = freq / max(1, len(tokens))

    # Adaptive weighting: only based on full concatenation
    if full_score > 0.85:
        # concatenation-mode
        final = (
            first_last  * 0.10 +
            set_ratio   * 0.25 +
            all_parts   * 0.10 +
            token_freq  * 0.05 +
            full_score  * 0.50
        )
    else:
        # traditional
        final = (
            first_last  * 0.15 +
            set_ratio   * 0.40 +
            all_parts   * 0.20 +
            token_freq  * 0.25
        )
    return final


def approximate_receiver_name(ocr_text, candidate_names, noise_words,
                              threshold=0.8, match_threshold=0.36):
    words = extract_ocr_words(ocr_text)
    cleaned = remove_noise_words(words, noise_words, threshold)

    scores = []
    for cand in candidate_names:
        score = score_candidate(cand, cleaned, threshold)
        scores.append((cand, score))
    top = nlargest(5, scores, key=lambda x: x[1])

    # Print the top 5 candidates with their scores
    print("Top 5 candidates:")
    for i, (candidate, score) in enumerate(top, 1):
        print(f"{i}. {candidate[0]} (Score: {score:.4f})")

    if top and top[0][1] >= match_threshold:
        return top[0]
    return (None, 0)

# List of known noise words (addresses, numbers, etc.)
noise_words = [
    "ni","no","resi","calvin",
    "institute","technology","menara","lt","rmci","jl","industri","blok",
    "b14","kav","kemayoran","pusat","10610","pademangan","kota","administrasi","utara",
    "dki","14410","jakarta","barat","express","tower","tangerang","jalan","alamat",
    "timur","spx","cit","penerima","pengirim","shopee","sameday","jkt","berat","ongkir","kg",
    "reguler",'tokopedia', 'sicepat', 'reg', 'sto', 'std', 'sp', 'central', 'pesanan', 'variasi', 'sku', 'produk',
    "rw", 'east', 'cod', 'cek', 'dulu', 'tidak', 'batas', 'kirim', "biok", "str", 'anteraja', 'oleh', 'murah',
    "shipping", "label", "gtl", "cllining", "wabehause", "class", "taaaattatatata", "drigin", "sorting",
    "tahcsit", "cit", "incino", "vanause", "it", "ship", "to", "address", "ip", "tracking", "numbelt",
    "intusosoaimpua", "ybil", "numbel", "fdt"
]
import random
import re
from heapq import nlargest
from rapidfuzz import fuzz

random.seed(42)

def extract_ocr_words(ocr_text):
    words = []
    for line in ocr_text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(',', 1)
        if parts and parts[0].strip():
            words.append(parts[0].strip().lower())
    return words


def remove_noise_words(words, noise_words, threshold=0.8):
    cleaned = []
    for w in words:
        if not any(fuzz.ratio(w, n)/100.0 > threshold for n in noise_words):
            cleaned.append(w)
    return cleaned


def fuzzy_token_set_ratio(name, words):
    tokens = name.lower().split()
    if not tokens:
        return 0.0
    total = 0.0
    for t in tokens:
        best = max((fuzz.ratio(t, w)/100.0 for w in words), default=0.0)
        total += best
    return total / len(tokens)


def full_name_concatenation_score(candidate, words):
    name = candidate[0].lower().replace(' ', '')
    if len(name) < 6:
        return 0.0
    best = 0.0
    for w in words:
        if w == name:
            return 1.0
        score = fuzz.ratio(w, name)/100.0
        best = max(best, score)
    return best


def score_candidate_detailed(candidate, words, threshold=0.8):
    detailed = {}
    tokens = candidate[0].lower().split()
    if not tokens:
        detailed['final_score'] = 0.0
        return detailed

    # full concat
    full_score = full_name_concatenation_score(candidate, words)
    detailed['full_concat_score'] = full_score

    # first & last
    first, last = tokens[0], tokens[-1]
    first_r = max((fuzz.ratio(first, w)/100.0 for w in words), default=0.0)
    last_r = max((fuzz.ratio(last,  w)/100.0 for w in words), default=0.0)
    detailed['first_token_ratio'] = first_r
    detailed['last_token_ratio'] = last_r
    if first_r > threshold and last_r > threshold:
        base = 1.0
    elif first_r > threshold or last_r > threshold:
        base = 0.7
    else:
        base = 0.0
    detailed['first_last_base_score'] = base
    bonus = 0.0
    mids_ratios = []
    if len(tokens) > 2:
        mids = tokens[1:-1]
        for m in mids:
            r = max((fuzz.ratio(m, w)/100.0 for w in words), default=0.0)
            mids_ratios.append(r)
            if r > threshold:
                bonus += (r / len(mids)) * 0.3
    detailed['middle_token_ratios'] = mids_ratios
    detailed['middle_token_bonus'] = bonus
    first_last = min(1.0, base + bonus)
    detailed['first_last_score'] = first_last

    # token set ratio
    set_ratio = fuzzy_token_set_ratio(candidate[0], words)
    detailed['token_set_ratio_score'] = set_ratio

    # all parts
    part_ratios = [max((fuzz.ratio(t, w)/100.0 for w in words), default=0.0) for t in tokens]
    detailed['part_ratios'] = part_ratios
    matched = sum(1 for pr in part_ratios if pr > threshold)
    detailed['matched_parts'] = matched
    all_parts = matched / len(tokens)
    detailed['all_parts_score'] = all_parts

    # token freq
    freq = sum(1 for t in tokens for w in words if t == w)
    token_freq = freq / len(tokens)
    detailed['token_freq'] = freq
    detailed['token_freq_score'] = token_freq

    # weights & final
    if full_score > 0.85:
        detailed.update({
            'weight_mode': 'concatenation',
            'first_last_weight': 0.10,
            'token_set_ratio_weight': 0.25,
            'all_parts_weight': 0.10,
            'token_freq_weight': 0.05,
            'full_concat_weight': 0.50
        })
        final = (first_last * 0.10 + set_ratio * 0.25 + all_parts * 0.10 + token_freq * 0.05 + full_score * 0.50)
    else:
        detailed.update({
            'weight_mode': 'traditional',
            'first_last_weight': 0.15,
            'token_set_ratio_weight': 0.40,
            'all_parts_weight': 0.20,
            'token_freq_weight': 0.25,
            'full_concat_weight': 0.00
        })
        final = (first_last * 0.15 + set_ratio * 0.40 + all_parts * 0.20 + token_freq * 0.25)

    detailed['final_score'] = final
    return detailed


def approximate_receiver_name(ocr_text, candidate_names, noise_words,
                              threshold=0.8, match_threshold=0.36):
    """
    Returns (best_candidate, best_score, detailed_scores) always.
    """
    # Extract and clean words
    words = extract_ocr_words(ocr_text)
    cleaned = remove_noise_words(words, noise_words, threshold)

    detailed_scores = {
        'original_words': words,
        'cleaned_words': cleaned,
        'candidate_scores': []
    }

    # Score all candidates
    scores = []
    for cand in candidate_names:
        det = score_candidate_detailed(cand, cleaned, threshold)
        scores.append((cand, det['final_score']))
        det_record = {
            'name': cand[0],
            'floor': cand[1],
            'final_score': det['final_score'],
            'detailed_scores': det
        }
        detailed_scores['candidate_scores'].append(det_record)

    # Select top
    top = nlargest(5, scores, key=lambda x: x[1])
    detailed_scores['top_matches'] = [
        {'name': c[0][0], 'floor': c[0][1], 'score': s} for c, s in top
    ]

    best_cand, best_score = top[0] if top else (None, 0.0)
    if best_score < match_threshold:
        best_cand = None

    return best_cand, best_score, detailed_scores

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
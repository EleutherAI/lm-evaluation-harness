import re
try:
    import jiwer
except ImportError:
   raise ImportError(
       "Please, install the `jiwer` package to use this module.\n" + 
       "You can install it using 'pip install jiwer'."
    )


def cer(items): return items
def der(items): return items


def agg_cer(items):
    avg_cer = 0
    refs, hyps = zip(*items)
    for ref, hyp in zip(refs, hyps):
        avg_cer += jiwer.cer(reference=ref, hypothesis=hyp)
    return f"{avg_cer / len(items) * 100}%"


def agg_der(items):
    # diacritics like kasra, fatha, shadda, ...etc.
    diacritics_pattern = re.compile(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
    )
    avg_der = 0
    refs, hyps = zip(*items)
    for ref, hyp in zip(refs, hyps):
        ref_diacritics = ' '.join([
            ''.join(diacritics_pattern.findall(word))
            for word in ref.split()
        ])
        hyp_diacritics = ' '.join([
            ''.join(diacritics_pattern.findall(word))
            for word in hyp.split()
        ])
        avg_der += jiwer.cer(reference=ref_diacritics, hypothesis=hyp_diacritics)
    return f"{avg_der / len(items) * 100}%"

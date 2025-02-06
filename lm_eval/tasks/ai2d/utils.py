import string


def flexible_extract(resps, docs):
    def filter_set(inst):
        filtered = []
        for resp in inst:
            while resp[-1] in string.punctuation:
                resp = resp[:-1]
            if resp[-1] in ["A", "B", "C", "D"]:
                resp = resp[-1]
            filtered.append(resp)
        return filtered

    filtered_resps = list(map(lambda x: filter_set(x), resps))

    return filtered_resps

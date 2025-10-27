def generate_k_values(max_k, num_values=6):
    candidates = [m * 10**p for p in range(10) for m in [1, 5] if m * 10**p <= max_k]
    return sorted(set(candidates[:num_values] + [max_k // 2, max_k]))

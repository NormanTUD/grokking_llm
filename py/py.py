import numpy as np

def get_frequencies(P, key_freqs):
    """Get angular frequencies w_k = 2*k*pi/P for key frequencies."""
    return [2 * k * np.pi / P for k in key_freqs]

def embed(token, P, key_freqs):
    """Map input token to sin/cos at key frequencies."""
    wks = get_frequencies(P, key_freqs)
    return [(np.cos(wk * token), np.sin(wk * token)) for wk in wks]

def combine_trig(embed_a, embed_b):
    """Compute cos(wk(a+b)) and sin(wk(a+b)) using trig identities."""
    combined = []
    for (cos_a, sin_a), (cos_b, sin_b) in zip(embed_a, embed_b):
        cos_sum = cos_a * cos_b - sin_a * sin_b  # cos(wk(a+b))
        sin_sum = sin_a * cos_b + cos_a * sin_b  # sin(wk(a+b))
        combined.append((cos_sum, sin_sum))
    return combined

def compute_logits(combined, P, key_freqs):
    """Compute logit for each c via cos(wk(a+b-c)), sum over frequencies."""
    wks = get_frequencies(P, key_freqs)
    logits = np.zeros(P)
    for c in range(P):
        for (cos_ab, sin_ab), wk in zip(combined, wks):
            # cos(wk(a+b-c)) = cos(wk(a+b))cos(wk*c) + sin(wk(a+b))sin(wk*c)
            logits[c] += cos_ab * np.cos(wk * c) + sin_ab * np.sin(wk * c)
    return logits

def modular_add(a, b, P=113, key_freqs=(14, 35, 41, 42, 52)):
    """Predict (a + b) mod P using the Fourier multiplication algorithm."""
    embed_a = embed(a, P, key_freqs)
    embed_b = embed(b, P, key_freqs)
    combined = combine_trig(embed_a, embed_b)
    logits = compute_logits(combined, P, key_freqs)
    return np.argmax(logits)

# Test
if __name__ == "__main__":
    P = 113
    correct = 0
    total = 200
    for _ in range(total):
        a, b = np.random.randint(0, P, size=2)
        pred = modular_add(a, b, P)
        if pred == (a + b) % P:
            correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")

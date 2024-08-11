def classify(x, threshold=0.5):
    return (x > threshold).astype(int)

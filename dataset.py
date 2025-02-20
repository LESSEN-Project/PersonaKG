from datasets import load_dataset
from difflib import SequenceMatcher

def similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def merge_sequences(sequences, threshold=0.8):

    merged_sequences = []
    used_indices = set()

    for i, seq1 in enumerate(sequences):
        if i in used_indices:
            continue
        merged = False
        for j in range(i + 1, len(sequences)):
            if j not in used_indices and similar(seq1, sequences[j], threshold):
                merged_sequence = merge_individual_sequences(seq1, sequences[j])
                merged_sequences.append(merged_sequence)
                used_indices.update([i, j])
                merged = True
                break
        if not merged:
            merged_sequences.append(seq1)
            used_indices.add(i)

    return merged_sequences

def merge_individual_sequences(seq1, seq2):

    sentences1 = seq1.split("\n")
    sentences2 = seq2.split("\n")

    merged_sentences = []
    used_indices = set()

    for s1 in sentences1:
        found_similar = False
        for idx, s2 in enumerate(sentences2):
            if idx not in used_indices and similar(s1, s2):
                merged_sentences.append(s1)
                used_indices.add(idx)
                found_similar = True
                break
        if not found_similar:
            merged_sentences.append(s1)

    for idx, s2 in enumerate(sentences2):
        if idx not in used_indices:
            merged_sentences.append(s2)

    return "\n".join(merged_sentences)

def get_dataset(split="train"):
    return load_dataset("google/Synthetic-Persona-Chat")["train"]

def get_personas(dataset, merge=True, threshold=0.6):

    merged_personas = dataset["user 1 personas"] + dataset["user 2 personas"]
    if merge:
        merged_personas = merge_sequences(merged_personas, threshold)

    return merged_personas
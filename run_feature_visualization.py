import matplotlib.pyplot as plt
import numpy as np
import torch

from ArticulatoryTextFrontend import ArticulatoryTextFrontend


def visualize_one_hot_encoded_sequence(tensor, sentence, col_labels, cmap='BuGn'):
    """
    Visualize a 2D one-hot encoded tensor as a heatmap.
    """
    tensor = torch.clamp(tensor, min=0, max=1).transpose(0, 1).cpu().numpy()
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be a 2D array")

    # Check the size of labels matches the tensor dimensions
    row_labels = ["stressed", "very-high-tone", "high-tone", "mid-tone", "low-tone", "very-low-tone", "rising-tone", "falling-tone", "peaking-tone", "dipping-tone", "lengthened", "half-length", "shortened", "consonant", "vowel", "phoneme", "silence", "end of sentence", "questionmark", "exclamationmark", "fullstop", "word-boundary", "dental", "postalveolar",
                  "velar", "palatal", "glottal", "uvular", "labiodental", "labial-velar", "alveolar", "bilabial", "alveolopalatal", "retroflex", "pharyngal", "epiglottal", "central", "back", "front_central", "front", "central_back", "mid", "close-mid", "close", "open-mid", "close_close-mid", "open-mid_open", "open", "rounded", "unrounded", "plosive",
                  "nasal", "approximant", "trill", "flap", "fricative", "lateral-approximant", "implosive", "vibrant", "click", "ejective", "aspirated", "unvoiced", "voiced"]

    if row_labels and len(row_labels) != tensor.shape[0]:
        raise ValueError("Number of row labels must match the number of rows in the tensor")
    if col_labels and len(col_labels) != tensor.shape[1]:
        raise ValueError("Number of column labels must match the number of columns in the tensor")

    plt.figure(figsize=(10, 8))

    # Create the heatmap
    plt.imshow(tensor, cmap=cmap, aspect='auto')

    # Add labels
    if row_labels:
        plt.yticks(np.arange(tensor.shape[0]), row_labels)
    if col_labels:
        plt.xticks(np.arange(tensor.shape[1]), col_labels, rotation=0)

    plt.grid(False)
    plt.xlabel('Phones')
    plt.ylabel('Features')

    # Display the heatmap
    plt.title(f"»{sentence}«")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sentence = "Rằng: Trong Thánh trạch dồi dào."
    language = "vie"

    tf = ArticulatoryTextFrontend(language=language)
    features = tf.string_to_tensor(sentence)
    phones = tf.get_phone_string(sentence)

    visualize_one_hot_encoded_sequence(tensor=features, sentence=sentence, col_labels=phones)

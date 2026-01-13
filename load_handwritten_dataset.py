from datasets import load_dataset


dataset = load_dataset("corto-ai/handwritten-text")
dataset.save_to_disk("./dataset")
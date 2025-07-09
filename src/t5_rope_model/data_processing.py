import random
from datasets import IterableDataset


def chunk_and_tokenize_stream(dataset: IterableDataset, tokenizer, chunk_size=512):
    token_buffer = []
    for example in dataset:
        # Filter out empty or whitespace-only documents on the fly
        text = example.get('text', '').strip()
        if not text:
            continue

        # Tokenize the text without special tokens
        tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        token_buffer.extend(tokens)

        # Yield chunks of chunk_size from the buffer
        while len(token_buffer) >= chunk_size:
            chunk = token_buffer[:chunk_size]
            yield {"input_ids": chunk}
            # Move the buffer forward
            token_buffer = token_buffer[chunk_size:]


def preprocess_for_t5_denoising(examples, tokenizer, corruption_rate=0.15, mean_noise_span_length=3.0):
    extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]
    extra_id_token_ids = tokenizer.convert_tokens_to_ids(extra_id_tokens)

    all_input_ids = []
    all_labels = []

    for input_ids in examples["input_ids"]:
        num_tokens = len(input_ids)
        num_to_corrupt = int(num_tokens * corruption_rate)

        corrupted_indices = set()
        # Create noise spans
        while len(corrupted_indices) < num_to_corrupt:
            # Sample a span length from an exponential distribution
            span_length = min(int(random.expovariate(1.0 / mean_noise_span_length)) + 1, 10)
            if span_length == 0: continue

            start_index = random.randint(0, num_tokens - span_length)
            # Add indices from the selected span to the set of corrupted indices
            corrupted_indices.update(range(start_index, start_index + span_length))

        # Ensure we don't exceed the number of tokens to corrupt
        corrupted_indices = sorted(list(corrupted_indices))[:num_to_corrupt]

        if not corrupted_indices:
            continue

        # Find continuous spans to replace with a single sentinel token
        spans = []
        current_span = []
        for i, index in enumerate(corrupted_indices):
            if not current_span or index == current_span[-1] + 1:
                current_span.append(index)
            else:
                spans.append(current_span)
                current_span = [index]
        if current_span:
            spans.append(current_span)

        # Limit to the number of available sentinel tokens
        if len(spans) > len(extra_id_token_ids):
            spans = spans[:len(extra_id_token_ids)]

        new_input_ids = []
        target_ids = []
        current_extra_id_idx = 0
        last_index = 0

        for span in spans:
            new_input_ids.extend(input_ids[last_index:span[0]])
            new_input_ids.append(extra_id_token_ids[current_extra_id_idx])
            target_ids.append(extra_id_token_ids[current_extra_id_idx])
            target_ids.extend([input_ids[i] for i in span])
            current_extra_id_idx += 1
            last_index = span[-1] + 1

        new_input_ids.extend(input_ids[last_index:])
        target_ids.append(tokenizer.eos_token_id)

        all_input_ids.append(new_input_ids)
        all_labels.append(target_ids)

    # Padding is now handled by the Trainer's data collator
    return {"input_ids": all_input_ids, "labels": all_labels}
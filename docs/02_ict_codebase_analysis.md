# In-Context Tuning (ICT) Codebase Analysis

## 1. Overview

**In-Context Tuning (ICT)** implements "Meta-learning via Language Model In-context Tuning" (Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, He He — ACL 2022). It recasts **few-shot learning** as a **sequence prediction problem**: K labeled demonstration examples are concatenated with a query example into a single input, and a language model is fine-tuned (meta-trained) to predict the query's label by "reading" the in-context demonstrations.

**Key Insight**: After meta-training across many tasks, the model becomes naturally better at in-context learning — it can generalize to **unseen tasks** at test time using the same demonstration-concatenation format, **without any gradient updates**.

**Relevance to Time-Series**: The same principle — conditioning on labeled examples at inference time without weight updates — can be adapted from NLP classification to time-series forecasting, as described in "In-Context Fine-Tuning for Time-Series Foundation Models."

---

## 2. Algorithm Design

### 2.1 Core Principle

ICT bridges three paradigms:

| Paradigm | How it works | ICT's relationship |
|----------|-------------|-------------------|
| **Prompting** (GPT-3 style) | Prepend examples to query, no training | ICT uses the same input format |
| **Fine-tuning** | Update model weights on task data | ICT meta-trains to be better at prompting |
| **Meta-learning** (MAML) | Learn to adapt via inner/outer loops | ICT achieves adaptation via in-context reading, not inner-loop gradients |

### 2.2 Two-Phase Algorithm

#### Phase 1: Meta-Training

For each training task (with its own examples, templates, and verbalizers):

1. **Sample demonstrations**: Randomly select K labeled examples from the task's support set (excluding the query)
2. **Construct input string**: Apply a randomly selected template to each demonstration (filling in the verbalizer for the label), then append the query with `[MASK]` replacing the label
3. **Forward pass**: Run the LM, extract logits at the `[MASK]` position, slice to verbalizer token IDs only
4. **Loss**: Cross-entropy between verbalizer logits and ground-truth label
5. **Update**: Full LM fine-tuning with AdamW + linear warmup schedule

The model learns to **read demonstrations and extract patterns** that inform the query prediction.

#### Phase 2: Meta-Testing (Zero-Shot on New Tasks)

For each test task (never seen during training):

1. Same input construction: K demos + query with `[MASK]`
2. Forward pass (no gradients)
3. Rank verbalizer logits to predict the query's label
4. **Multiple random demo samplings** (`num_prefix_selections`) to reduce variance from demonstration selection

### 2.3 Input Format

```
[demo_1_with_label] [delimiter] [demo_2_with_label] [delimiter] ... [demo_K_with_label] [delimiter] [query_with_MASK]
```

Example (LAMA knowledge probing):
```
Paris is the capital of France . Berlin is the capital of Germany . Tokyo is the capital of [MASK] .
```

---

## 3. Codebase Structure

```
In-context/
  src/
    ict.py              # Main orchestrator: meta-train and meta-test loops
    verbalized_model.py # LM wrapper: verbalizer-based classification
    data_loader.py      # Demonstration sampling, template filling, input construction
    example.py          # Example usage script
  example_data/
    class_verbalizers.txt  # ~21K single-token verbalizer vocabulary
  example/
    5shot_fold0/
      model.pkl            # Pretrained ICT checkpoint
```

---

## 4. Key Classes

### 4.1 `ICT` Class (`ict.py`)

The top-level orchestrator managing the full pipeline.

**Meta-Training (`meta_train` method):**
```python
def meta_train(self):
    for epoch in range(num_epochs):
        # Build all batches across all training tasks
        all_batches = []
        for task in train_tasks:
            task_examples = self.data_loader.prepare_all_inputs(task, ...)
            task_batches = chunk_into_batches(task_examples, bsz)
            all_batches.extend(task_batches)
        random.shuffle(all_batches)

        for batch in all_batches:
            loss = self.model(input_ids, attention_mask, labels)
            scaler.scale(loss).backward()  # fp16 mixed precision
            optimizer.step()
            scheduler.step()
```

**Meta-Testing (`meta_test` method):**
```python
def meta_test(self):
    for task in test_tasks:
        for query in task_examples:
            all_logits = []
            for _ in range(num_prefix_selections):  # Multiple random demo sets
                demos = sample_demonstrations(query, task, K)
                input_str = construct_input(demos, query)
                logits = self.model.predict(input_str)
                all_logits.append(logits)
            avg_logits = mean(all_logits)
            predicted_label = argmax(avg_logits)
```

**Key Design Decisions:**
- **Per-task batching**: All examples in a batch come from the same task (share verbalizer token IDs)
- **Cross-task shuffling**: Batches from different tasks are shuffled across training
- **fp16 training**: `torch.cuda.amp.GradScaler` for memory efficiency
- **Full LM fine-tuning**: All parameters are updated (no frozen layers)

### 4.2 `VerbalizedModel` Class (`verbalized_model.py`)

Wraps a HuggingFace language model for verbalizer-based classification.

**Core Mechanism:**
```python
class VerbalizedModel:
    def __init__(self, model_name, task_format):
        if task_format == 'mlm':
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif task_format == 'clm':
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, verbalizer_word_ids, labels=None):
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits  # [B, seq_len, vocab_size]

        if self.task_format == 'mlm':
            # Find [MASK] position in each input
            mask_positions = (input_ids == mask_token_id).nonzero()
            mask_logits = logits[batch_idx, mask_pos, :]  # [B, vocab_size]
        elif self.task_format == 'clm':
            mask_logits = logits[:, -1, :]  # Last position logits

        # Slice to verbalizer tokens only
        class_logits = mask_logits[:, verbalizer_word_ids]  # [B, num_classes]

        if labels is not None:
            loss = CrossEntropyLoss(class_logits, labels)
            return loss
        return class_logits
```

**Key Insight — Verbalizer Mapping**: Instead of a classification head, the model classifies by predicting which **verbalizer token** the LM assigns highest probability at the `[MASK]` position. Each class maps to a single token (e.g., class 0 → "#", class 1 → "%"). The constraint is that **each verbalizer must tokenize to exactly one token**.

### 4.3 `Data_loader` Class (`data_loader.py`)

Handles all data processing: demonstration sampling, template filling, and input construction.

**Data Structures:**

| Structure | Type | Description |
|---|---|---|
| `task2examples` | `Dict[str, List[Dict]]` | Task name → list of examples. Each example = dict with input fields + `<label>` (int) |
| `task2templates` | `Dict[str, List[str]]` | Task name → list of template strings with placeholders (`<sub>`, `<obj>`, `<label>`) |
| `task2verbalizers` | `Dict[str, List[str]]` | Task name → list of verbalizer words (one per class) |

**Key Methods:**

```python
def sample_demonstrations(self, query, task_examples, K):
    """Randomly sample K examples from task, excluding query itself"""
    candidates = [ex for ex in task_examples if not is_same(ex, query)]
    if not allow_label_overlap:
        candidates = [ex for ex in candidates if ex['<label>'] != query['<label>']]
    return random.sample(candidates, K)

def encode_example_with_template(self, example, template):
    """Fill template with example fields"""
    # Labeled version: "Paris is the capital of France"
    labeled = template.replace('<sub>', example['<sub>']).replace('<label>', verbalizer[label])
    # Query version:   "Paris is the capital of [MASK]"
    query_ver = template.replace('<sub>', example['<sub>']).replace('<label>', '[MASK]')
    return labeled, query_ver

def encode_input_str(self, labeled_demos, masked_query, delimiter):
    """Concatenate: [demo1] [delim] [demo2] [delim] ... [demoK] [delim] [query]"""
    return delimiter.join(labeled_demos + [masked_query])
```

**Demonstration Selection Strategy:**
- **Random sampling** (not nearest-neighbor or sophisticated selection)
- **No self-inclusion**: Query always excluded from demo candidates
- **Label diversity**: `allow_label_overlap=False` forces demos to cover different classes
- **Variance reduction at test time**: `num_prefix_selections` random demo sets, results averaged

---

## 5. Hyperparameters & Configuration

| Parameter | Default Value | Notes |
|---|---|---|
| `model_name` | `bert-base-cased` | Any HuggingFace MLM/CLM model |
| `task_format` | `mlm` | `mlm` (BERT) or `clm` (GPT) |
| `num_demonstrations` | 5 | K-shot demonstrations |
| `example_delimiter` | `" "` (space) | Separator between concatenated examples |
| `allow_label_overlap` | `False` | Whether demos can share query's label |
| `lr` | `3e-6` | Learning rate |
| `num_warmup_steps` | `100` | Linear warmup steps |
| `num_epochs` | `15` | Meta-training epochs |
| `bsz` | `48` | Batch size (per-task) |
| `num_prefix_selections` | `20` | Random demo sets at test time for averaging |

---

## 6. Training Details

### 6.1 Optimization

- **Optimizer**: AdamW with configurable learning rate
- **Scheduler**: Linear warmup + linear decay (via `get_linear_schedule_with_warmup`)
- **Mixed precision**: fp16 via `torch.cuda.amp.GradScaler` (always enabled)
- **Full fine-tuning**: All LM parameters updated (asserted `requires_grad=True` for all params)

### 6.2 Loss Function

Standard **Cross-Entropy** on verbalizer logits:
```python
loss = CrossEntropyLoss(class_logits, ground_truth_label_index)
# class_logits: [B, num_classes] (sliced from vocab logits)
```

### 6.3 Evaluation Metrics

- **Precision@1**: Fraction of queries where top-1 prediction is correct
- **Precision@10**: Fraction where correct answer is in top-10
- **MRR**: Mean Reciprocal Rank of the correct verbalizer among all verbalizers

---

## 7. Key Algorithmic Properties

### 7.1 Why ICT Works Without Gradient Updates at Test Time

During meta-training, the model learns a **generalizable skill**: reading K labeled demonstrations and extracting task-relevant patterns to predict the query's label. This skill transfers to unseen tasks because:

1. The concatenation format is consistent across training and testing
2. The model learns to use **positional relationships** between demo labels and demo inputs
3. The verbalizer vocabulary provides a universal output space

### 7.2 Relationship to Few-Shot Learning

| Method | Inner Loop | Outer Loop | Test-Time Adaptation |
|--------|-----------|------------|---------------------|
| MAML | Gradient steps per task | Meta-optimize initial weights | Few gradient steps |
| Prompting | None | None (pretrained LM) | Zero-shot via demos |
| **ICT** | None (demos as input) | Meta-train LM on many tasks | **Zero-shot via demos** |

ICT achieves the adaptation benefits of MAML without per-task gradient computation, and the simplicity of prompting with better robustness to demo selection/ordering.

### 7.3 Scalability Considerations

| Factor | Impact |
|--------|--------|
| K (num demos) | Input length grows linearly: `seq_len ≈ K × avg_demo_len + query_len` |
| Attention cost | Quadratic in sequence length: `O((K+1)² × d)` |
| Model size | Larger models generally benefit more from ICT |
| Verbalizer quality | Single-token constraint limits expressiveness; critical for performance |

---

## 8. Transferable Concepts for Time-Series Adaptation

The following ICT principles are directly applicable to time-series forecasting:

1. **Demonstration-augmented inference**: Prepend K (history, future) pairs as context to the query — the model can leverage observed future patterns from demos to inform predictions
2. **No weight updates**: A well-pretrained foundation model can adapt to unseen domains purely through in-context conditioning
3. **Variance reduction via ensemble**: Multiple random demo selections averaged at test time stabilize predictions
4. **Demo selection matters**: Random sampling is a baseline; task-aware selection (e.g., temporally or spatially similar demos) may improve performance

**Key difference from NLP**: In time-series, the "verbalizer" concept doesn't apply. Instead, the "label" in a demonstration is the **ground-truth future values** — continuous, high-dimensional, and directly usable as attention values in the model's cross-attention mechanism.

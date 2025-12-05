# LLM fine-tuning and performance improvement: a model-centric approach

This document presents the strategy adopted to complete the second laboratory of the Scalable Machine Learning and Deep Learning course at KTH. In particular, we focused on improving the performance of the fine-tuned Llama-3.2-1B model by adopting a **model-centric approach** as required by Task 2.1(a). Although the data-centric perspective is equally valuable, we chose to work on hyperparameter optimization to better understand how different training configurations and LoRA settings influence model performance.

Besides this main task, this README contains the following sections:
* Checkpointing methodology
* Model-centric improvement
* Creative UI to provide value to stakeholders
* Useful links

# Checkpointing methodology

To implement the checkpointing functionality, as required in Task 1, to restart the training from where we left off, we decided to mount a Google Drive folder and save weights every 100 steps. This configuration was done inside the training configuration function `SFTTrainer` in the part reserved for `TrainingArguments`. 

After that, before starting the training, our implementation checks if any checkpoint is present in the folder `/content/drive/MyDrive/ID2223_Lab2/checkpoints` of our personal drive; if it's the case, the training restarts from the latest checkpoint. Moreover, for safety and reliability purposes, we decided to save the last 2 checkpoints. Most relevant code for this part is reported below:

```
from google.colab import drive
drive.mount('/content/drive')
checkpoint_dir = "/content/drive/MyDrive/ID2223_Lab2/checkpoints"
...
trainer = SFTTrainer(
    ...
    args = TrainingArguments(
        ...
        save_strategy = "steps",  # Enable checkpointing
        save_steps = 100,         # Save every 100 steps
        save_total_limit = 2,     # Save last 2 checkpoint
        output_dir = checkpoint_dir, #checkpoint_dir, # Salva i checkpoint su Drive
    ),
)
...
import os
import glob

# Check if any checkpoints exist in the output directory

output_dir = trainer.args.output_dir
checkpoints = list(glob.glob(os.path.join(output_dir, "checkpoint-*")))

if len(checkpoints) > 0:
    print(f"Found {len(checkpoints)} checkpoints. Resuming from the latest one.")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("No checkpoints found. Starting new training.")
trainer_stats = trainer.train()
```

# Model-centric improvement

## Baseline model configuration

To measure improvement, we established as a baseline our first configuration of the Llama-3.2-1B model in the provided, which can be retrieved in the ``LLM_finetuning-LLAMA-3.2-1B.ipynb`` notebook. The configuration is the following:

| *Parameter* | *Baseline value* | *Rationale* | 
| :--- | :--- | :--- | 
| Foundation Model | Llama 3.2 1B | Chosen for fast inference on CPU environments. | 
| Optimizer | AdamW | Standard for modern transformer training. | 
| Learning Rate (lr) | **2e-4** | Standard starting point for QLoRA. | 
| LoRA Rank (r) | **16** | A lower rank to minimize parameter count and memory usage. | 
| Metric | Training Loss | Used to monitor generalization during training. | 

## Model-centric improvement experiments

We conducted two distinct experiments focusing on core model parameters to drive performance improvement.

### Experiment 1: Learning rate (LR) tuning

The Learning Rate is the most critical hyperparameter. The starting value (2e-4) can be too high, leading to unstable training, oscillations in the loss function, and potentially preventing the model from converging to the global optimum.
So, we decided to systematically lower it from 2e-4 to a more cautious **5e-6**.

#### *Hypothesis*

A lower learning rate allows the optimizer to take smaller and more precise steps. This stabilizes the training process, leading to a lower final training loss and better generalization. 

We used the training loss statistics provided in the original notebook.

#### *Results of improvement*

The experiment resulted in a significant reduction in the final training loss.

| *Run* | *Learning rate* | *Final Training Loss* | 
| :--- | :--- | :--- | 
| *Baseline version* | **2e-5** | 0.854 | 
| *Improved version* | **5e-6** | 0.698 |

#### *Conclusion*

The reduced LR (5e-6) achieved a reduction in training loss compared to the baseline, indicating superior convergence and better generalization.

### Experiment 2: LoRA rank (r) tuning

Secondly, we decided to work with the rank tuning, a fundamental parameter in LoRA technology since LoRA (Low-Rank Adaptation) leverages the concept of lower-rank matrices to make the model training process extremely efficient and fast. A higher rank allows the adapter to approximate the necessary weight changes more accurately.

The LoRA Rank (r) was increased from the baseline of **16** to a higher capacity of **64**. To maintain the necessary scaling ratio (alpha/r = 2), lora_alpha was increased to **128*.

#### *Hypothesis*

By increasing the LoRA rank (r), the model gains greater capacity to capture the complexity present in the dataset, this, of course, requires more time in training.

#### *Results of improvement *

We measured the improvement considering the training time (per 1000 samples) and the training loss, obtaining the following:

| *Run* | *LoRA Rank (r)* | *Training Time* | *Traning loss* | 
| :--- | :--- | :--- | :--- | 
| *Baseline* | *16* | 73 min | 0.854 | 
| *Improvement* | *64* | 97 min | *0.752* | 

#### *Conclusion* 

Increasing the LoRA rank to 64 slightly increased training time (due to more trainable parameters) but resulted in a better training loss. This demonstrates that the higher capacity adapter captured the task-specific patterns better, leading to significantly higher quality and relevance in the generated text.

## Further model-centric improvements

*Base Model Selection:* A further improvement could be to change the model from Llama-3.2-1B to a larger, but still CPU-friendly model, like *Mistral 7B* (quantized to 4-bit) or another Llama variant, accepting a small trade-off in inference speed for a major boost in reasoning ability.

## Conclusion

As final model, we decided to keep the baseline one, since it's a good compromise between performance and computational time. So, we deployed it on Hugging Face after training with 5000 samples. Unfortunately, we were not able to process the entire dataset due to Colab restrictions on GPU usage and long times required for resource reallocation.

# The user interface for continuous update

Our users can interact with the model through an *Hugging Face* space publicly exposed. This application is not limited to model querying, but it also includes a feedback collection mechanism.

## Chat with the LLM

The core of the demo has been created with *Gradio*'s `ChatInterface` library, a simple and efficient way to build a chatbot.

## Collect User Impressions

The project specification required a **creative improvement** to the basic UI to deliver more value to the stakeholders. Our group decided to embed a powerful flagging collection that allows for future model evaluations and adaptations to user needs. *Gradio* usually embeds three ways for doing so:

* Flagging Modes
* Callback Functions
* Local Storage

However, **none of them is available** when the application is deployed on *Hugging Face*. For this reason, we needed to implement our own code, which throws additional `Gradio Blocks` allowing the users to comment on model behaviours and push this precious information into a *Hugging Face* dataset.

## Extensive Feedback

The stakeholders can do much more than a simple thumb-up thumb-down selection, as they have a wide range of motivations to select among. For this reason, in a real-world scenario, the *EVALS* dataset we have built would be an important insight to capture future user needs and improvements, as well as a tool to ensure the **social fairness** of the model.

# Useful Links

This *GitHub* repository contains a copy of the Colab code we used for finetuning and a copy of the Gradio application for the model demo. To see the actual deployment, you are encouraged to follow these links:

* *Hugging face* space to chat with the model: [https://huggingface.co/spaces/fedealex/chat](https://huggingface.co/spaces/fedealex/chat)
* *Hugging face* model containing the final LLM: [https://huggingface.co/fedealex/llama-1B/tree/main](https://huggingface.co/fedealex/llama-1B/tree/main)
* *Hugging face* dataset for the evaluation metrics: [https://huggingface.co/datasets/fedealex/flags/tree/main](https://huggingface.co/datasets/fedealex/flags/tree/main)

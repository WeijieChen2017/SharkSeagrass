# Logging Models and Code with Weights & Biases (wandb)

In this guide, we'll demonstrate how to set up logging with Weights & Biases (wandb) to track your experiments, models, and code. This process involves installing the `wandb` package, initializing a run, logging your code, logging the model and optimizer state dictionaries, and loading these artifacts for continued training.

## Step 1: Install wandb

Make sure you have the `wandb` package installed. You can install it via pip if you haven't already:

```bash
pip install wandb
```

## Step 2: Initialize wandb in Your Training Script

To start logging, you need to login to wandb using your personal API key and initialize a run. Insert the following code snippet at the beginning of your training script, ideally after your imports and before your training loop:

```python
import wandb
import os

# Login to wandb with your personal API key
wandb.login(key="personal_identification_key")

# Initialize a new wandb run
wandb_run = wandb.init(
    project="CT_ViT_VQGAN",
    dir=os.getenv("WANDB_DIR", "cache/wandb"), 
    config=global_config  # Ensure global_config is defined in your script
)
```

### Explanation of Parameters

- project: This is the name of the project under which your runs will be grouped. Make sure it is descriptive and unique to your specific project.
- dir: The directory where wandb will store run data locally. By default, it uses os.getenv("WANDB_DIR", "cache/wandb"). This means it will look for the environment variable WANDB_DIR and use its value if set; otherwise, it defaults to cache/wandb.
- config: This is where you pass your configuration settings, typically a dictionary containing hyperparameters and other run-specific settings.

### Example Configuration

Here's an example of how you might define and use global_config in your script:

```python
global_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    # Add other hyperparameters and settings here
}

wandb_run = wandb.init(
    project="CT_ViT_VQGAN",
    dir=os.getenv("WANDB_DIR", "cache/wandb"), 
    config=global_config
)
```

## Step 3: Log Your Code

To keep track of the exact version of the code used for each run, you can log your code with wandb. This is especially useful for ensuring reproducibility and tracking changes over time. Insert the following line in your script:
    
```python
# Log the current version of the code
wandb_run.log_code(root=".", name="train.py")
```

### Explanation of Parameters

- root: The root directory to look for code files. The default value is the current directory ".".
name: The specific file or pattern to log. In this example, we log train.py, but you can specify other files or use patterns to log multiple files.

## Step 4: Log Model and Optimizer State Dict

To log the model and optimizer state dictionaries, use the following code snippet at the appropriate place in your training script, typically after you save the model and optimizer state_dict:

```python
# Define the save names for the model and optimizer
model_save_name = save_folder + f"model_best_{idx_epoch}_state_dict_{current_level}.pth"
optimizer_save_name = save_folder + f"optimizer_best_{idx_epoch}_state_dict_{current_level}.pth"

# Save the model and optimizer state_dict
torch.save(model.state_dict(), model_save_name)
torch.save(optimizer.state_dict(), optimizer_save_name)

# Log the model and optimizer state_dict to wandb
wandb_run.log_model(path=model_save_name, name="model_best_eval", aliases=f"{current_level}")
wandb_run.log_model(path=optimizer_save_name, name="optimizer_best_eval", aliases=f"{current_level}")
```

### Explanation of Parameters

- path: The path to the file you want to log.
- name: A name to identify the logged file.
- aliases: A list of tags to associate with the logged file, which can be useful for tracking different versions or states.

## Step 5: Load Model and Optimizer for Continued Training

To continue training from the saved state, initialize wandb in your continuation training script and load the model and optimizer state dictionaries as follows:

### Initialize wandb

```python
import wandb
import os

# Login to wandb with your personal API key
wandb.login(key="personal_identification_key")

# Initialize a new wandb run
wandb_run = wandb.init(
    project="CT_ViT_VQGAN",
    job_type="try_load_artifact"
)
```

### Load Artifacts

```python
model_artifact_name = "model_best_eval:v79"
optim_artifact_name = "optimizer_best_eval:v79"

for artifact_name in [model_artifact_name, optim_artifact_name]:
    artifact = wandb_run.use_artifact(f"convez376/CT_ViT_VQGAN/{artifact_name}")
    artifact_dir = artifact.download()
```

### Load Model and Optimizer State Dicts

```python
import glob
import torch

state_dict_model_path = glob.glob("./artifacts/" + model_artifact_name + "/*.pth")[0]
state_dict_optim_path = glob.glob("./artifacts/" + optim_artifact_name + "/*.pth")[0]

print(state_dict_model_path)
print(state_dict_optim_path)

state_dict_model = torch.load(state_dict_model_path)
state_dict_optim = torch.load(state_dict_optim_path)

# Print keys for verification
for key in state_dict_model.keys():
    print(key)
print("=" * 30)
for key in state_dict_optim.keys():
    print(key)
```

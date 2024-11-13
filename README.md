# itp450imageclassification
Final Project for ITP-450. Distributed ML model training on HPC/SLURM using multiple GPUs and Nodes


Summary of files

<h2> DDP_CIFAR10_Model.py </h2>
Initial CNN model implemented with DDP

<h2> DDP_CIFAR10.job </h2>
this job file starts a job running DDP_CIFAR10_Model.py

<h2> itp450_final_project.ipynb </h2>
Playground for testing models and workflows

<h2> Light_CIFAR10_Pytorch_Lightning.py </h2>
Model programmed using PyTorch Lightning for maintainability <br />
This model is light enough to run locally on your laptop.
Does not use DDP and

<h2> Lightning_DDP.py </h2>
Model programmed using Torch Lightning, large model and 
uses DDP through the Lightning Trainer API.

<h2> train.job </h2>
Empty so far, I presume I would make a script to run a Lightning_DDP job

If you wanna learn, quick links to docs
<a href="https://lightning.ai/docs/pytorch/1.6.2/common/trainer.html"> Lighting Trainer API </a>
<a href="https://lightning.ai/docs/pytorch/1.6.2/common/lightning_module.html"> Lighting Module API </a>
<a href="https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html#"> Lighting Tensor Parallelism </a>
<a href="https://lightning.ai/docs/pytorch/1.6.2/accelerators/gpu.html"> Multi GPU Training </a>

Some more Random stuff as reference
<a href="https://pytorch.org/docs/stable/distributed.tensor.parallel.html"> Parallelization with Vanilla Torch </a>
<a href="https://pytorch.org/tutorials/intermediate/TP_tutorial.html"> Large Scale Transformer training with vanilla torch </a>
<a href="https://pytorch.org/tutorials/intermediate/dist_tuto.html"> Writing Distributed models with torch </a>
<a href="https://pytorch.org/tutorials/intermediate/ddp_tutorial.html"> Torch tutorials </a>
<a href="https://pytorch.org/tutorials/intermediate/dist_tuto.html"> Writing Distributed models with torch </a>
<a href="https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#llm-example"> 
    Tensor Parallelism with Lightning AI
</a>


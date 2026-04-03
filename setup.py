from setuptools import setup, find_packages

setup(
    name="v-mla-qwen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "triton>=2.1.0",
    ],
    author="Your Name",
    description="V-MLA: Visual Multi-head Latent Attention for Qwen 3.5-VL",
    python_requires=">=3.9",
)

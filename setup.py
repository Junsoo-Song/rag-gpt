from setuptools import setup, find_packages

setup(
    name="rag-gpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "langchain>=0.1.0",
        "langchain-community",
        "langchain-groq",
        "langchain-huggingface",
        "faiss-cpu",
        "pypdf",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "rag-gpt=rag_gpt.__main__:app",
        ],
    },
    python_requires=">=3.8",
)


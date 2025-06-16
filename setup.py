from setuptools import setup, find_packages
import subprocess
import sys

# Install spaCy model
def install_spacy_model():
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing spaCy model: {e}")
        sys.exit(1)

setup(
    name="query-optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform>=1.92.0",
        "google-cloud-storage>=2.19.0",
        "google-genai>=1.14.0",
        "gitpython>=3.1.40",
        "google-adk>=0.5.0",
        "sentence-transformers>=2.5.1",
        "faiss-cpu>=1.7.4",
        "numpy>=1.26.4",
        "python-dotenv>=1.0.1",
        "tqdm>=4.66.2",
        "pypdf>=4.1.0",
        "python-docx>=1.1.0",
        "gdown>=4.7.3",
        "spacy>=3.7.2",
    ],
    python_requires=">=3.8",
)

if __name__ == "__main__":
    install_spacy_model() 
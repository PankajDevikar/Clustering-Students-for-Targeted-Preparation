from setuptools import setup, find_packages

setup(
    name="student_clustering_app",
    version="1.0.0",
    author="Pankaj Devikar",
    author_email="pankaj.devikar@gmail.com",
    description="A Streamlit app for clustering students based on performance and learning styles.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit==1.25.0",
        "pandas==2.1.3",
        "numpy==1.26.1",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.1",
        "seaborn==0.12.3",
    ],
    entry_points={
        "console_scripts": [
            "student-clustering=app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

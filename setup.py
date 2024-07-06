from setuptools import setup, find_packages

setup(
    name="DetectoBuddy",
    version="0.1.0",
    author="AR10Dev & LF-D3v",
    description="An advanced object detection application to identify objects in images, videos, and live webcam feeds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AR10Dev/DetectoBuddy",
    packages=find_packages(),  # Automatically find and include all packages
    # install_requires=[
    #     # List your project's dependencies here.
    #     # Examples:
    #     # 'numpy>=1.18.1',
    #     # 'pandas==1.0.3',
    # ],
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    # entry_points={
    #     'console_scripts': [
    #         # This allows you to create executable scripts.
    #         # Example:
    #         # 'script_name = module_name:function_name',
    #     ],
    # },
)

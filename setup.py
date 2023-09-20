from setuptools import setup, find_packages

setup(
    name='manipulator_mujoco',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # List the packages and versions from requirements.txt
        line.strip() for line in open('requirements.txt')
    ],
)

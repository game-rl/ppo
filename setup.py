from setuptools import setup, find_namespace_packages

setup(
    name='gamerl.ppo.jax',
    version='0.1.0',
    packages=find_namespace_packages(
        include=['gamerl.ppo.jax'],
    ),
    install_requires=[],
    author='pi-tau',
    url='https://github.com/game-rl/ppo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

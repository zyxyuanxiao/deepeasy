from setuptools import setup


def load_requirements(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f]


setup(
    name='deepeasy',
    version='0.9',
    author='zzzzer',
    author_email='zzzzer91@gmail.com',
    url='https://github.com/zzzzer91/deepeasy',
    packages=['deepeasy'],
    install_requires=load_requirements('./requirements.txt'),
    zip_safe=False,
)

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements from requirements.txt
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove new line characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Ignore the '-e .' trigger during the list generation
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='NE-EPICGUARD',
    version='0.0.1',
    author='Data Scientist',
    author_email='pal_sumeetkumar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
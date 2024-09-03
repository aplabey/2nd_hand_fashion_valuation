from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='2nd_hand_fashion_valuation',
      version="0.0.12",
      description="2nd_hand_fashion_valuation Model (api_pred)",
    #  license=<>,
    #  author="",
    #  author_email="",
      #url="https://github.com/aplabey/2nd_hand_fashion_valuation",
      install_requires=requirements,
      packages=find_packages(),
    #  test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)

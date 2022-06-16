import SM

VERSION = SM.__version__
DISTNAME = "soil moisture module"
DESCRIPTION = ("Soil Moisture module for Machine Learning Cafe Project Soil Moisture",)
AUTHORS = "Timo Houben, Johannes Boog, Swamini Khurana, Lennart Schmidt, Julia Schmid, Pia Ebeling, Mohit Anand"
EMAIL = "timo.hoube@ufz.de"
LICENSE = "Copyright(c) 2005-2020, Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ. All rights reserved."
PROJECT_URLS = {
    "GitLab": "https://git.ufz.de/ml-cafe/ml-cafe_project_soilmoisture/-/tree/master/SM_module",
    "Documentation": "TBDone",
}


from setuptools import setup, find_packages

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    authors=AUTHORS,
    author_email=EMAIL,
    packages=find_packages(exclude=["tests*", "docs*"]),
    license=LICENSE,
)

import SM

VERSION = SM.__version__
DISTNAME = "soil moisture module"
DESCRIPTION = ("Soil moisture module for digital soil moisture mapping.",)
AUTHORS = "Timo Houben, Swamini Khurana, Johannes Boog, Julia Schmid, Pia Ebeling, Mohit Anand, Lennart Schmidt"
EMAIL = "timo.hoube@ufz.de"
LICENSE = "Copyright(c) 2005-2020, Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ. All rights reserved."
PROJECT_URLS = {
    "GitHub": "https://github.com/timohouben/sm-module",
}


from setuptools import setup

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    authors=AUTHORS,
    author_email=EMAIL,
    license=LICENSE,
)

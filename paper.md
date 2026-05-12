---
title: "ACTUNEO: An open-source Python library for actuarial modelling with an African market focus"
tags:
  - actuarial science
  - insurance
  - pensions
  - mortality
  - financial mathematics
  - Africa
authors:
  - name: Shannon Tafadzwa Sikadi
    affiliation: 1
affiliations:
  - name: Independent actuarial developer (Zimbabwe)
    index: 1
date: 2026-05-12
bibliography: paper.bib
---

# Summary

ACTUNEO is an open-source Python library that provides reusable building blocks for actuarial computations used in insurance and pension analytics, with particular attention to the data and workflow realities common in African markets. The library focuses on methods that recur across actuarial work: mortality table and survival function calculations, interest theory, yield curve construction and interpolation, duration/convexity measures, and life-contingent present value calculations for life assurance, annuities, and reserves. ACTUNEO is designed to integrate naturally into modern scientific Python workflows by using standard data structures and interoperating with NumPy and pandas for numerical work and tabular analysis [@harris2020array; @mckinney2010data].

The project is distributed as a Python package, is documented with an online Sphinx site, and includes automated tests and continuous integration. By providing a coherent actuarial toolkit that is easy to install, test, and extend, ACTUNEO aims to support reproducible actuarial research and education as well as transparent, reviewable analytical pipelines in applied settings.

# Statement of need

Many actuarial analyses require combining several types of calculations—mortality modelling, discounting and yield curve work, and life-contingent valuation—into a single reproducible workflow. In practice, these workflows are frequently implemented as spreadsheets or ad-hoc scripts, which can be hard to test, hard to review, and difficult to reuse across studies. While there are actuarial packages available across languages and ecosystems, the available tooling is often fragmented across domains or packaged in a way that is less accessible to teams that are standardising on the scientific Python stack.

ACTUNEO addresses this need by offering a modular, Python-native set of actuarial components that can be composed in research code and teaching materials. In addition to a general actuarial scope, ACTUNEO explicitly targets practical friction points encountered in African and Zimbabwean contexts, such as the need to work with local mortality, inflation, and interest rate assumptions, multi-currency environments, and regulatory reporting conventions. The library’s modular structure separates domain components (e.g., mortality, finance, life contingencies) so that users can adopt a small subset or extend the package with additional market- or regulator-specific layers without rewriting core numerical routines.

The intended users are researchers, students, and practitioners who want actuarial functionality inside a version-controlled, testable Python workflow. Typical research uses include sensitivity analyses of life expectancy and survival functions, reproducible evaluation of life-contingent cashflow models under alternative discount curves, and the construction of benchmark calculations to validate methods implemented in downstream models.

# State of the field

Python is widely used for applied data analysis and research software development, and actuarial workflows in Python can leverage mature numerical and data libraries such as NumPy and pandas [@harris2020array; @mckinney2010data], with SciPy providing numerical routines used in many scientific applications [@virtanen2020scipy]. Within actuarial and related quantitative finance communities, several libraries exist that cover subsets of life contingencies and financial mathematics. For example, *actuarialmath* provides educational implementations of life actuarial mathematics in Python [@actuarialmath], and *lifecontingencies* provides life contingencies functionality in R [@lifecontingencies]. These projects are valuable references and are used by practitioners and educators.

ACTUNEO complements this landscape by packaging a set of actuarial methods as a cohesive, extensible library intended for integration into broader scientific Python workflows and documentation-first learning materials. A distinguishing focus of ACTUNEO is to make it straightforward to incorporate assumptions and data relevant to African markets in a transparent way, while keeping the underlying computations general and reusable. In this sense, the project aims to reduce duplicated “local glue code” that is often re-implemented in private spreadsheets or scripts and to promote reproducible, reviewable actuarial research artifacts.

# Implementation and design

ACTUNEO is organised into modules aligned with common actuarial domains. The current implementation includes a mortality module (mortality tables and survival functions), a finance module (interest theory, yield curve interpolation, duration/convexity), and a life module (life assurance, annuities, and reserves). The package emphasises deterministic, testable computations and exposes functions and classes that can be composed in downstream analyses. The project also includes scaffolding for additional domains (e.g., pensions, IFRS 17 tooling, loss reserving, macroeconomic connectors, simulation); these are intended as extension points for future work and community contributions rather than as fully implemented features in the current release.

To support usability and sustainability, ACTUNEO is distributed using standard Python packaging (PEP 621 metadata), includes automated tests, and provides a Sphinx documentation site with API references and worked examples. Continuous integration runs linting and tests across multiple Python versions to help ensure consistent behaviour.

# Availability

The software is licensed under the MIT license and is publicly available at https://github.com/ShannonT20/ACTUNEO. The package can be installed from PyPI as `actuneo`.

# Acknowledgements

ACTUNEO is inspired by prior actuarial open-source projects and by the broader scientific Python ecosystem. The author thanks the actuarial community for feedback and discussions that shaped the initial scope and design decisions.

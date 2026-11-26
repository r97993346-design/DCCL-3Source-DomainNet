# PACS · Adam 1e-5 learning-rate diagnosis

Complete observations available 7.

First five trial IDs 0, 1, 2, 3, 4.
Positive late slope means the metric rose between steps 3000 and 4999.

| Metric | Step 0 | Step 1500 | Step 3000 | Step 4999 | Late slope per 1000 updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total loss | 4.02396 | 0.570073 | 0.408314 | 0.314209 | -0.0470762 |
| E classification | 0.405174 | 0.0601836 | 0.0412091 | 0.0299748 | -0.00561999 |
| S uniform | 2.7063 | 0.025378 | 0.00831053 | 0.00271385 | -0.00279974 |
| CondCov | 0.013079 | 0.00896702 | 0.0074729 | 0.00622454 | -0.000624492 |
| RD | 0.426628 | 0.1905 | 0.112734 | 0.0623628 | -0.0251982 |
| CLIP contrastive | 4.30731 | 2.98723 | 2.65966 | 2.50597 | -0.0768878 |
| Source validation cross-entropy (tr_outloss) | 0.380349 | 0.0557926 | 0.0442722 | 0.0406694 | -0.0018023 |
| Source validation accuracy | 0.905958 | 0.983752 | 0.985128 | 0.985706 | 0.000288993 |

Selected checkpoint histogram 3000 (1 folds), 3900 (6 folds), 4200 (4 folds), 4500 (4 folds), 4800 (4 folds), 4999 (1 folds).

| Sample | Gradient norm | Actual Adam update norm | Relative update |
| --- | ---: | ---: | ---: |
| 0 | 63.21 | 0.0073038 | 0.00022793 |
| 300 | 6.0083 | 0.0032407 | 0.00010121 |
| 900 | 5.0732 | 0.0017931 | 5.5965e-05 |
| 1800 | 4.3294 | 0.001496 | 4.6698e-05 |
| 3000 | 4.0268 | 0.0014078 | 4.3913e-05 |
| 4200 | 3.6994 | 0.0014117 | 4.4182e-05 |
| 4800 | 4.115 | 0.001404 | 4.3979e-05 |
| 4999 | 3.3175 | 0.0016779 | 5.2383e-05 |

| Feature-gradient pair | Median cosine | Fraction below −0.1 | Samples |
| --- | ---: | ---: | ---: |
| CondCov vs E classification on E | 0.076 | 0.0% | 160 |
| CondCov vs RD on E | 0.254 | 0.0% | 160 |
| CondCov vs S uniform on S | 0.007 | 0.0% | 160 |

Late checkpoint selections 65%. Mean late source validation gain +0.058 pp. Mean late source cross-entropy decrease 8.14%.
Median late source-validation reversals of at least 0.2 pp 0.0.
These five observations support keeping 1e-5 as the reference for a later comparison.

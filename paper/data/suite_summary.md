# Competition suite summary (strict grading, avg@k over problems; problem-weighted pools)

| arm | aime_2025 (30) | aime_2026 (30) | brumo_2025 (30) | cmimc_2025 (40) | hmmt_feb_2025 (30) | smt_2025 (53) | aime_2022 (30) | aime_2023 (30) | aime_2024 (30) | amc_2022_2023 (83) | hmmt_feb_2023 (30) | hmmt_feb_2024 (30) | math500 (500) | post-cutoff | pre-2025 | competition | all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_chat | 7.5 | - | - | - | 2.1 | - | - | - | 10.0 | - | - | 3.3 | 71.0 | 4.8 (n=60) | 6.7 (n=60) | 5.7 (n=120) | 58.4 (n=620) |
| c256het_bitrev | 27.9 | - | - | - | - | - | - | - | 34.2 | - | - | - | 85.6 | 27.9 (n=30) | 34.2 (n=30) | 31.0 (n=60) | 79.8 (n=560) |
| c256tpl_bitrev | 31.2 | 35.0 | 44.2 | 16.6 | 24.6 | 38.0 | 37.5 | 32.1 | 41.2 | 72.6 | 27.9 | 21.7 | 84.8 | 31.6 (n=213) | 46.5 (n=233) | 39.4 (n=446) | 63.4 (n=946) |
| c256uni_bitrev | 24.6 | - | - | - | - | - | - | - | 40.8 | - | - | - | 86.8 | 24.6 (n=30) | 40.8 (n=30) | 32.7 (n=60) | 81.0 (n=560) |
| c256uni_u128 | 27.1 | 33.8 | 44.6 | 14.4 | 22.9 | 35.4 | 34.6 | 28.8 | 40.8 | 69.9 | 26.7 | 19.6 | 85.8 | 29.6 (n=213) | 44.3 (n=233) | 37.2 (n=446) | 62.9 (n=946) |
| c256vph_bitrev | 26.2 | - | - | - | - | - | - | - | 37.1 | - | - | - | 84.2 | 26.2 (n=30) | 37.1 (n=30) | 31.7 (n=60) | 78.6 (n=560) |
| vanilla_canonical | 32.9 | 36.7 | 43.3 | 19.4 | 22.1 | 42.0 | 37.9 | 36.7 | 44.2 | 73.0 | 32.5 | 24.6 | 87.4 | 33.1 (n=213) | 48.7 (n=233) | 41.2 (n=446) | 65.6 (n=946) |

## Paired per-problem bootstrap of the difference (points of avg@k, 95% CI, common problems only)

- base_chat vs vanilla_canonical: post_cutoff: -22.7 [-30.8, -14.4] n=60; pre_2025: -27.7 [-36.7, -19.2] n=60; competition: -25.2 [-31.6, -19.5] n=120; all: -18.1 [-21.3, -14.9] n=620
- base_chat vs c256uni_u128: post_cutoff: -20.2 [-28.8, -11.9] n=60; pre_2025: -23.5 [-31.7, -16.0] n=60; competition: -21.9 [-27.7, -16.5] n=120; all: -16.2 [-19.4, -13.0] n=620
- c256het_bitrev vs vanilla_canonical: post_cutoff: -5.0 [-13.3, +3.3] n=30; pre_2025: -10.0 [-16.2, -4.2] n=30; competition: -7.5 [-12.7, -2.5] n=60; all: -2.4 [-5.0, +0.1] n=560
- c256het_bitrev vs c256uni_u128: post_cutoff: +0.8 [-3.8, +5.4] n=30; pre_2025: -6.7 [-13.8, +0.0] n=30; competition: -2.9 [-6.9, +1.2] n=60; all: -0.5 [-3.1, +1.9] n=560
- c256tpl_bitrev vs vanilla_canonical: post_cutoff: -1.5 [-3.8, +0.9] n=213; pre_2025: -2.1 [-4.1, -0.1] n=233; competition: -1.8 [-3.4, -0.3] n=446; all: -2.2 [-3.8, -0.6] n=946
- c256tpl_bitrev vs c256uni_u128: post_cutoff: +2.0 [+0.2, +3.8] n=213; pre_2025: +2.3 [+0.4, +4.2] n=233; competition: +2.1 [+0.8, +3.5] n=446; all: +0.5 [-1.0, +2.0] n=946
- c256uni_bitrev vs vanilla_canonical: post_cutoff: -8.3 [-15.8, -1.7] n=30; pre_2025: -3.3 [-9.6, +2.1] n=30; competition: -5.8 [-10.8, -1.2] n=60; all: -1.2 [-3.4, +1.2] n=560
- c256uni_bitrev vs c256uni_u128: post_cutoff: -2.5 [-5.4, +0.4] n=30; pre_2025: +0.0 [-5.4, +5.0] n=30; competition: -1.2 [-4.2, +1.7] n=60; all: +0.8 [-1.5, +3.0] n=560
- c256uni_u128 vs vanilla_canonical: post_cutoff: -3.5 [-5.8, -1.3] n=213; pre_2025: -4.4 [-6.3, -2.4] n=233; competition: -4.0 [-5.5, -2.5] n=446; all: -2.7 [-4.3, -1.3] n=946
- c256vph_bitrev vs vanilla_canonical: post_cutoff: -6.7 [-15.0, +0.4] n=30; pre_2025: -7.1 [-13.3, -0.4] n=30; competition: -6.9 [-12.1, -2.1] n=60; all: -3.6 [-6.0, -1.2] n=560
- c256vph_bitrev vs c256uni_u128: post_cutoff: -0.8 [-5.0, +3.8] n=30; pre_2025: -3.8 [-10.4, +2.1] n=30; competition: -2.3 [-5.8, +1.5] n=60; all: -1.7 [-4.3, +0.7] n=560
- vanilla_canonical vs c256uni_u128: post_cutoff: +3.5 [+1.3, +5.8] n=213; pre_2025: +4.4 [+2.4, +6.3] n=233; competition: +4.0 [+2.5, +5.5] n=446; all: +2.7 [+1.3, +4.3] n=946

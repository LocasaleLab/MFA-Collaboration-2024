from scripts.src.core.common.config import CoreConstants
natural_c13_ratio = CoreConstants.natural_c13_ratio

glucose_6_input_metabolite_dict = {
    "GLC_e": [
        {
            "ratio_list":
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            "abundance": 1,
        },
    ],
}

glucose_2_input_metabolite_dict = {
    "GLC_e": [
        {
            "ratio_list":
                [
                    1,
                    1,
                    natural_c13_ratio,
                    natural_c13_ratio,
                    natural_c13_ratio,
                    natural_c13_ratio,
                ],
            "abundance": 1,
        },
    ],
}

glutamine_5_input_metabolite_dict = {
    "GLN_e": [
        {
            "ratio_list":
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            "abundance": 1,
        },
    ],
}


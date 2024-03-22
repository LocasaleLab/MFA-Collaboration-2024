from scripts.src.core.common.config import CoreConstants
from scripts.src.core.data.data_class import InputMetaboliteData


class Direct(object):
    data_direct = 'scripts/data'


class DataType(object):
    susan_data = 'susan_data'
    fangchao_fly_data = 'fangchao_fly_data'
    yimon_data = 'yimon_data'
    fangchao_cultured_cell_data = 'fangchao_cultured_cell_data'
    lianfeng_data = 'lianfeng_data'


glucose_6_labeled_input_metabolite_dict = {
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


glutamine_5_labeled_input_metabolite_dict = {
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


pyruvate_3_labeled_input_metabolite_dict = {
    "PYR_e": [
        {
            "ratio_list":
                [
                    1,
                    1,
                    1,
                ],
            "abundance": 1,
        },
    ],
}


glucose_unlabeled_ratio_list = [CoreConstants.natural_c13_ratio] * 6


def input_mid_data_processor(input_raw_metabolite_dict):
    return {
        input_metabolite_name: InputMetaboliteData(input_metabolite_name, abundance_data_list)
        for input_metabolite_name, abundance_data_list in input_raw_metabolite_dict.items()}


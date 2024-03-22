from common_and_plotting_functions.functions import check_and_mkdir_of_direct
from scripts.src.common.config import Color, Keywords, Direct
from scripts.data.config import DataType
from scripts.src.common.plotting_functions import group_bar_plot, multi_row_col_bar_plot
from scripts.src.common.functions import data_param_list_generator_func_template, collect_results_func_template
from scripts.src.common.result_processing_functions import common_flux_comparison_func, \
    experimental_data_plotting_func_template

from scripts.data.common_functions import common_data_loader

from scripts.model.model_loader import model_loader, ModelList
from scripts.src.core.model.model_constructor import common_model_constructor

data_wrap_obj, keyword = common_data_loader(DataType.fangchao_fly_data, test_mode=False, natural_anti_correction=False)
base_model = model_loader(ModelList.infusion_model)
# base_model = model_loader(ModelList.base_model_with_glc_buffer)
mfa_model_obj = common_model_constructor(base_model)


class SpecificParameter(object):
    test_dynamic_constant_flux_list = []
    test_preset_constant_flux_value_dict = {
        'GLC_total_input': 100,
        # 'GLC_supplement_net': 0,
    }
    specific_flux_range_dict = {
        'Salvage_c': (1, 10),
    }


separate_data_param_raw_list = [
    {
        keyword.labeling: keyword.sucrose,
        '': [
            {
                keyword.tissue: keyword.whole_body,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            {
                keyword.tissue: keyword.sperm,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
        ]
    },
    {
        keyword.labeling: keyword.glutamine,
        '': [
            {
                keyword.tissue: keyword.whole_body,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            {
                keyword.tissue: keyword.sperm,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: 1,
                                    },
                                    {
                                        keyword.index: 2,
                                    },
                                    {
                                        keyword.index: 3,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
        ]
    }
]

average_data_param_raw_list = [
    {
        keyword.labeling: keyword.sucrose,
        '': [
            {
                keyword.tissue: keyword.whole_body,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            {
                keyword.tissue: keyword.sperm,
                '': [
                    {
                        keyword.age: keyword.adult,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        keyword.age: keyword.old,
                        '': [
                            {
                                keyword.condition: keyword.ctrl,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                            {
                                keyword.condition: keyword.mr_fa,
                                '': [
                                    {
                                        keyword.index: Keywords.average,
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
        ]
    },
]

important_flux_list = [
    ('FBA_c', 'FBA_c__R'),
    ('PGM_c', 'PGM_c__R'),
    'Salvage_c',
    'PC_m',
    'G6PDH2R_PGL_GND_c',
    'PDH_m',
    'CS_m',
    'AKGD_m',
    'ICDH_m',
    'PYK_c',
    'ACITL_c',
    ('SUCD_m', 'SUCD_m__R'),
    ('FUMH_m', 'FUMH_m__R'),
    ('MDH_m', 'MDH_m__R'),
    ('GPT_c', 'GPT_c__R'),
    'PHGDH_PSAT_PSP_c',
    ('LDH_c', 'LDH_c__R'),
    'GLN_input',
    'ASP_input',
    ('SHMT_c', 'SHMT_c__R'),
    'GLC_input',
    'PEPCK_c',
    ('GAPD_c', 'GAPD_c__R'),
]

mid_name_list = [
    ['GLC_c', 'FRU6P_c+GLC6P_c', 'E4P_c'],
    ['2PG_c+3PG_c', 'PEP_c', 'PYR_c+PYR_m', 'LAC_c'],
    ['CIT_c+CIT_m+ICIT_m', 'AKG_c+AKG_m', 'GLU_c+GLU_m'],
    ['SUC_m', 'FUM_m', 'MAL_c+MAL_m', 'ASP_c+ASP_m']
]


# complete_data_param_raw_list = separate_data_param_raw_list
complete_data_param_raw_list = average_data_param_raw_list

data_param_raw_list = complete_data_param_raw_list
keyword_list = [keyword.labeling, keyword.tissue, keyword.age, keyword.condition, keyword.index]
total_param_list = data_param_list_generator_func_template(
    keyword_list)(data_param_raw_list)

collect_results = collect_results_func_template(
    data_wrap_obj.project_name_generator, total_param_list, keyword_list)


def flux_comparison_parameter_generator(final_solution_data_dict, final_flux_name_index_dict):
    from ...common.config import index_calculation_func_dict
    current_index_name_func_dict = index_calculation_func_dict
    final_dict_for_comparison = {}
    final_key_name_parameter_dict = {}
    final_color_dict = {}
    comparison_dict = {
        keyword.ctrl_mr_mrfa: [keyword.ctrl, keyword.mr, keyword.mr_fa]
    }
    labeling_list = [keyword.sucrose]
    tissue_list = [keyword.whole_body, keyword.sperm]
    age_list = [keyword.adult, keyword.old]
    current_important_flux_list = list(important_flux_list)
    if current_index_name_func_dict is not None:
        current_important_flux_list.extend(current_index_name_func_dict.items())
    for comparison_name, current_condition_list in comparison_dict.items():
        for labeling in labeling_list:
            for tissue in tissue_list:
                current_solution_data_dict = final_solution_data_dict[labeling][tissue]
                complete_comparison_name = f'{comparison_name}/{labeling}_{tissue}'
                key_name_parameter_dict = {}
                data_dict_for_plotting = {}
                color_dict = {}
                for age in age_list:
                    for condition_index, condition_name in enumerate(current_condition_list):
                        for index_num, current_data_array in \
                                current_solution_data_dict[age][condition_name].items():
                            key_name = f'{age}_{condition_name}_{index_num}'
                            key_name_parameter_dict[key_name] = (age, condition_name, index_num)
                            if condition_index == 0:
                                current_color = Color.blue
                            elif condition_index == 1:
                                current_color = Color.orange
                            else:
                                current_color = Color.purple
                            if key_name not in color_dict:
                                color_dict[key_name] = current_color
                            common_flux_comparison_func(
                                current_important_flux_list,
                                final_flux_name_index_dict[labeling][tissue][age][condition_name][index_num],
                                current_data_array, data_dict_for_plotting, key_name)
                final_dict_for_comparison[complete_comparison_name] = data_dict_for_plotting
                final_color_dict[complete_comparison_name] = color_dict
                final_key_name_parameter_dict[complete_comparison_name] = key_name_parameter_dict
    return final_dict_for_comparison, final_key_name_parameter_dict, final_color_dict


target_emu_name_nested_list = [
    ['glucose', 'dihydroxyacetone phosphate', 'pyruvate', 'lactate'],
    ['citrate/isocitrate', 'a-ketoglutarate', 'succinate', 'malate'],
    ['glutamate', 'glutamine', 'aspartate', 'methionine'],
]


def major_minor_key_analysis_func(result_information_dict):
    labeling = result_information_dict[keyword.labeling]
    tissue = result_information_dict[keyword.tissue]
    age = result_information_dict[keyword.age]
    condition = result_information_dict[keyword.condition]
    index = result_information_dict[keyword.index]
    labeling_tissue_key = f'{labeling}_{tissue}'
    age_condition_index_str = f'{age}_{condition}_{index}'
    major_key = labeling_tissue_key
    minor_key_list = [age, condition, index]
    minor_key_str = age_condition_index_str
    if condition == keyword.ctrl:
        current_color = Color.blue
    elif condition == keyword.mr:
        current_color = Color.orange
    elif condition == keyword.mr_fa:
        current_color = Color.purple
    else:
        raise ValueError()
    return major_key, minor_key_list, minor_key_str, current_color, 0


experimental_data_plotting = experimental_data_plotting_func_template(
    target_emu_name_nested_list, major_minor_key_analysis_func)


def metabolic_network_parameter_generator():
    experimental_mid_metabolite_set = {
        'GLC_c', 'FBP_c', 'DHAP_c', 'GAP_c', '3PG_c', 'PEP_c',
        'PYR_c', 'PYR_m', 'LAC_c', 'ALA_c', 'ERY4P_c',
        'CIT_m', 'MAL_m', 'AKG_m', 'SUC_m', 'ASP_m',
        'SER_c', 'GLY_c', 'ASP_c', 'CIT_c', 'MAL_c',
        'GLU_m', 'GLN_m', 'GLU_c', 'GLN_c', 'AKG_c', 'RIB5P_c', 'ERY4P'
    }
    experimental_mixed_mid_metabolite_set = {
        'PYR_c', 'PYR_m', 'CIT_m', 'CIT_c', 'MAL_m', 'MAL_c',
        'GLU_m', 'GLU_c', 'GLN_m', 'GLN_c', 'ASP_m', 'ASP_c',
        'AKG_m', 'AKG_c',
    }
    biomass_metabolite_set = {
        'ALA_c', 'RIB5P_c', 'GLY_c', 'SER_c', 'ASP_c',
        'ACCOA_c', 'GLU_c', 'GLN_c',
    }
    input_metabolite_set = {
        'GLC_e', 'GLC_unlabelled_e', 'GLN_e', 'ASP_e', 'SER_e', 'GLY_e', 'ALA_e', 'LAC_e',
    }
    c13_labeling_metabolite_set = {
        'GLC_e',
    }
    boundary_flux_set = {
        'GLC_input', 'GLC_unlabelled_input'
    }
    infusion = True
    return experimental_mid_metabolite_set, experimental_mixed_mid_metabolite_set, biomass_metabolite_set, \
        input_metabolite_set, c13_labeling_metabolite_set, boundary_flux_set, infusion

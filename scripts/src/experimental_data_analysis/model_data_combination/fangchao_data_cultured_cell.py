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

from .fangchao_data_fruit_fly import metabolic_network_parameter_generator as fly_metabolic_network, \
    important_flux_list, mid_name_list

data_wrap_obj, keyword = common_data_loader(
    DataType.fangchao_cultured_cell_data, test_mode=False, natural_anti_correction=False)
base_model = model_loader(ModelList.base_model_with_glc_tca_buffer)
mfa_model_obj = common_model_constructor(base_model)


class SpecificParameter(object):
    test_dynamic_constant_flux_list = []
    test_preset_constant_flux_value_dict = {
        'GLC_input': 100,
        'GLC_supplement_net': 0,
        'CIT_supplement_net': 0,
        # 'GLC_supplement_net': 0,
    }
    specific_flux_range_dict = {
        'Salvage_c': (1, 10),
    }


separate_data_param_raw_list = [
    {
        keyword.tissue: keyword.cell_293t,
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
        keyword.tissue: keyword.cell_gc1,
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
        keyword.tissue: keyword.cell_gc2,
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

average_data_param_raw_list = [
    {
        keyword.tissue: keyword.cell_293t,
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
        keyword.tissue: keyword.cell_gc1,
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
        keyword.tissue: keyword.cell_gc2,
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


def data_param_list_generator(param_raw_list):
    current_param_list = []
    for raw_param_dict in param_raw_list:
        tissue_key = raw_param_dict[keyword.tissue]
        each_tissue_content_list = raw_param_dict['']
        for tissue_content_dict in each_tissue_content_list:
            condition_key = tissue_content_dict[keyword.condition]
            each_condition_content_list = tissue_content_dict['']
            for condition_dict in each_condition_content_list:
                index_key = condition_dict[keyword.index]
                new_param_dict = {
                    keyword.tissue: tissue_key,
                    keyword.condition: condition_key,
                    keyword.index: index_key,
                }
                current_param_list.append(new_param_dict)
    return current_param_list


# complete_data_param_raw_list = separate_data_param_raw_list
complete_data_param_raw_list = average_data_param_raw_list
data_param_raw_list = complete_data_param_raw_list
keyword_list = [keyword.tissue, keyword.condition, keyword.index]
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
    tissue_list = [keyword.cell_293t, keyword.cell_gc1, keyword.cell_gc2]
    current_important_flux_list = list(important_flux_list)
    if current_index_name_func_dict is not None:
        current_important_flux_list.extend(current_index_name_func_dict.items())
    for comparison_name, current_condition_list in comparison_dict.items():
        for tissue in tissue_list:
            current_solution_data_dict = final_solution_data_dict[tissue]
            complete_comparison_name = f'{comparison_name}/{tissue}'
            key_name_parameter_dict = {}
            data_dict_for_plotting = {}
            color_dict = {}
            for condition_index, condition_name in enumerate(current_condition_list):
                for index_num, current_data_array in \
                        current_solution_data_dict[condition_name].items():
                    key_name = f'{condition_name}_{index_num}'
                    key_name_parameter_dict[key_name] = (condition_name, index_num)
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
                        final_flux_name_index_dict[tissue][condition_name][index_num],
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
    tissue = result_information_dict[keyword.tissue]
    condition = result_information_dict[keyword.condition]
    index = result_information_dict[keyword.index]
    age_condition_index_str = f'{condition}_{index}'
    major_key = tissue
    minor_key_list = [condition, index]
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
    (
        experimental_mid_metabolite_set, experimental_mixed_mid_metabolite_set, biomass_metabolite_set,
        input_metabolite_set, c13_labeling_metabolite_set, boundary_flux_set, _) = fly_metabolic_network()
    infusion = False
    return experimental_mid_metabolite_set, experimental_mixed_mid_metabolite_set, biomass_metabolite_set, \
        input_metabolite_set, c13_labeling_metabolite_set, boundary_flux_set, infusion

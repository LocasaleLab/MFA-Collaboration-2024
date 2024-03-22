import warnings

import numpy as np

from scripts.src.common.built_in_packages import it, defaultdict
from scripts.src.common.third_party_packages import pd
from scripts.src.common.config import DataType
from scripts.src.core.common.config import CoreConstants
from scripts.src.core.common.classes import TransformDict
from scripts.src.core.data.data_class import MIDData, MFAData, average_multiple_mid_data
from .dataset_inventory import return_dataset_and_keyword
from .config import input_mid_data_processor, glucose_6_labeled_input_metabolite_dict

debug = False


def target_metabolite_data_loader(
        xlsx_file_path, xlsx_sheet_name, mixed_compartment_list=('c', 'm'),
        index_col_name='Name', row_num=None, col_range=None, to_standard_name_dict=None,
        excluded_metabolite_name_set=None, specific_dimension_metabolite_dict=None):
    if col_range is not None:
        col_range = list(range(*col_range))
    if excluded_metabolite_name_set is None:
        excluded_metabolite_name_set = set()
    if to_standard_name_dict is None:
        to_standard_name_dict = TransformDict()
    if specific_dimension_metabolite_dict is None:
        specific_dimension_metabolite_dict = {}
    raw_data_frame = pd.read_excel(
        xlsx_file_path, sheet_name=xlsx_sheet_name, index_col=index_col_name, nrows=row_num, usecols=col_range)
    group_metabolite_row_dict = {}
    current_metabolite_name = ''
    current_metabolite_group_list = []
    for row_index, raw_metabolite_name in enumerate(raw_data_frame.index):
        if raw_metabolite_name is np.nan:
            continue
        raw_metabolite_name = raw_metabolite_name.strip('-+ ')
        if (
                raw_metabolite_name == current_metabolite_name or '13C' in raw_metabolite_name or
                ('-' in raw_metabolite_name and raw_metabolite_name[:raw_metabolite_name.rindex('-')]
                    == current_metabolite_name)):
            if '_13C' in raw_metabolite_name:
                # raw_metabolite_name = raw_metabolite_name.strip('-+ ')
                num_str_loc = raw_metabolite_name.rindex('_13C')
                num_str = raw_metabolite_name[num_str_loc + len('_13C'):]
                if not num_str.isdigit():
                    raise ValueError()
                else:
                    isotopomer_index = int(num_str)
                    while len(current_metabolite_group_list) <= isotopomer_index:
                        current_metabolite_group_list.append(-1)
                    current_metabolite_group_list[isotopomer_index] = row_index
            else:
                current_metabolite_group_list.append(row_index)
        else:
            group_metabolite_row_dict[current_metabolite_name] = current_metabolite_group_list
            if '-' in raw_metabolite_name:
                last_minus_loc = raw_metabolite_name.rindex('-')
                last_minus_part = raw_metabolite_name[last_minus_loc + 1:]
                if last_minus_part.isdigit() or last_minus_part.startswith('m+'):
                    current_metabolite_name = raw_metabolite_name[:last_minus_loc]
                else:
                    current_metabolite_name = raw_metabolite_name
            else:
                current_metabolite_name = raw_metabolite_name
            current_metabolite_group_list = [row_index]
    group_metabolite_row_dict[current_metabolite_name] = current_metabolite_group_list
    final_raw_data_dict = {}
    for condition_name in raw_data_frame.columns:
        current_data_dict = {}
        for raw_metabolite_name, metabolite_row_list in group_metabolite_row_dict.items():
            if raw_metabolite_name == '':
                continue
            standard_metabolite_name, combined_metabolite_list = standardize_metabolite_name(raw_metabolite_name)
            if standard_metabolite_name in excluded_metabolite_name_set:
                continue
            raw_data_list = []
            for metabolite_row in metabolite_row_list:
                if metabolite_row == -1:
                    current_raw_data_element = 0
                else:
                    current_raw_data_element = raw_data_frame[condition_name].iloc[metabolite_row]
                if current_raw_data_element is None or np.isnan(current_raw_data_element):
                    current_raw_data_element = 0
                raw_data_list.append(current_raw_data_element)
            if len(raw_data_list) == 1 or sum(raw_data_list) == 0:
                print(
                    f'[Data loader] Zero sum: Data from metabolite {raw_metabolite_name} '
                    f'in condition {condition_name} in sheet {xlsx_sheet_name} is left out')
                # raise ValueError(standard_metabolite_name)
                continue
            if standard_metabolite_name in specific_dimension_metabolite_dict:
                specific_dim_num = specific_dimension_metabolite_dict[standard_metabolite_name]
                while len(raw_data_list) < specific_dim_num:
                    raw_data_list.append(0)
            new_mid_data_obj = MIDData(
                raw_data_list=raw_data_list, raw_name=standard_metabolite_name,
                combined_raw_name_list=combined_metabolite_list, to_standard_name_dict=to_standard_name_dict,
                compartment_list=mixed_compartment_list)
            if new_mid_data_obj.normalize(CoreConstants.eps_for_mid):
                current_data_dict[new_mid_data_obj.name] = new_mid_data_obj
        final_raw_data_dict[condition_name] = current_data_dict
    return final_raw_data_dict


def standardize_metabolite_name(raw_name: str):
    standard_name = raw_name
    standard_name = standard_name.strip()
    standard_name = standard_name.lower()
    special_suffix_list = ['_pos', '_neg', '-neg']
    for common_metabolite_sep in CoreConstants.common_metabolite_sep_list:
        standard_name = standard_name.replace(common_metabolite_sep, CoreConstants.standard_metabolite_sep)
    if standard_name.endswith('-)'):
        standard_name = standard_name[:standard_name.rindex('(')]
    for special_suffix in special_suffix_list:
        if standard_name.endswith(special_suffix):
            standard_name = standard_name[:-len(special_suffix)]
            break
    # if standard_name.endswith('_pos'):
    #     standard_name = standard_name[:-len('_pos')]
    # if standard_name.endswith('_neg'):
    #     standard_name = standard_name[:-len('_neg')]
    # if standard_name.endswith('-neg'):
    #     standard_name = standard_name[:-len('-neg')]
    combined_metabolite_list = []
    if CoreConstants.standard_metabolite_sep in standard_name:
        combined_metabolite_list = [
            raw_metabolite_name.strip()
            for raw_metabolite_name in standard_name.split(CoreConstants.standard_metabolite_sep)]
    return standard_name, combined_metabolite_list


def glucose_6_input_metabolite_obj_generator():
    return input_mid_data_processor(glucose_6_labeled_input_metabolite_dict)


class DataWrap(object):
    def __init__(self, data_name, dataset_obj, input_metabolite_obj_data_dict=None):
        self.data_name = data_name
        self.dataset_obj = dataset_obj
        target_data_parameter_dict = dataset_obj.return_data_parameter_dict()
        for data_name, current_data_parameter_dict in target_data_parameter_dict.items():
            current_data_dict = target_metabolite_data_loader(**current_data_parameter_dict)
            dataset_obj.add_data_sheet(data_name, current_data_dict)
        self.default_input_metabolite_obj_data_dict = glucose_6_input_metabolite_obj_generator()
        self.ratio_dict_to_objective_func = defaultdict(lambda: 1)

    def project_name_generator(self, tissue_name, tissue_index, repeat_index, *args):
        return self.dataset_obj.project_name_generator(tissue_name, tissue_index, repeat_index, *args)

    def return_dataset(self, param_dict=None):
        (
            project_name, final_target_metabolite_data_dict,
            final_input_metabolite_data_dict, *combined_data_and_ratio) = self.dataset_obj.return_dataset(param_dict)
        if final_input_metabolite_data_dict is None:
            final_input_metabolite_data_dict = self.default_input_metabolite_obj_data_dict
        mfa_data_obj = MFAData(
            project_name, final_target_metabolite_data_dict, final_input_metabolite_data_dict,
            *combined_data_and_ratio)
        return mfa_data_obj


def common_data_loader(dataset_name, test_mode=False, natural_anti_correction=False):
    dataset_obj, keyword = return_dataset_and_keyword(dataset_name)
    dataset_obj.set_data_status(test_mode)
    dataset_obj.set_anti_correction(natural_anti_correction)
    data_wrap_obj = DataWrap(dataset_name, dataset_obj)
    return data_wrap_obj, keyword


def average_mid_data_dict(parent_raw_data_dict, index_average_list):
    final_target_metabolite_data_list_dict = {}
    for tmp_index in index_average_list:
        current_target_metabolite_data_dict = parent_raw_data_dict[tmp_index]
        for metabolite_name, mid_data_obj in current_target_metabolite_data_dict.items():
            if metabolite_name not in final_target_metabolite_data_list_dict:
                final_target_metabolite_data_list_dict[metabolite_name] = []
            final_target_metabolite_data_list_dict[metabolite_name].append(mid_data_obj)
    final_target_metabolite_data_dict = {}
    for metabolite_name, each_metabolite_mid_list in final_target_metabolite_data_list_dict.items():
        final_target_metabolite_data_dict[metabolite_name] = average_multiple_mid_data(each_metabolite_mid_list)
    return final_target_metabolite_data_dict

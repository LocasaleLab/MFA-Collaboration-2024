from .data_metabolite_to_standard_name_dict import data_metabolite_to_standard_name_dict
from .fly_specific_data_parameters import Keyword as BasicKeyword
from scripts.src.common.config import Direct, DataType, Keywords as CommonKeywords
from ..complete_dataset_class import CompleteDataset
from ..common_functions import average_mid_data_dict


class Keyword(BasicKeyword):
    cell_293t = '293t'
    cell_gc1 = 'gc1'
    cell_gc2 = 'gc2'


class SpecificParameters(CompleteDataset):
    def __init__(self):
        super(SpecificParameters, self).__init__()
        self.mixed_compartment_list = ('c', 'm')
        self.current_direct = "{}/fangchao_data".format(Direct.data_direct)
        self.file_path = "{}/13C_labeling_cultured_cell_data_Fangchao.xlsx".format(self.current_direct)
        self.sheet_name_dict = {
            '293T': {
                Keyword.tissue: Keyword.cell_293t,
            },
            'GC1': {
                Keyword.tissue: Keyword.cell_gc1,
            },
            'GC2': {
                Keyword.tissue: Keyword.cell_gc2,
            },
        }
        self.test_experiment_name_prefix = '293T'
        self.test_col = 'Ctrl_1'
        self._complete_data_parameter_dict_dict = {
            current_sheet_name: {
                'xlsx_file_path': self.file_path,
                'xlsx_sheet_name': current_sheet_name,
                'index_col_name': 'Name',
                'mixed_compartment_list': self.mixed_compartment_list,
                'to_standard_name_dict': data_metabolite_to_standard_name_dict}
            for current_sheet_name in self.sheet_name_dict.keys()}
        self._test_data_parameter_dict_dict = {
            DataType.test: {
                'xlsx_file_path': self.file_path,
                'xlsx_sheet_name': self.test_experiment_name_prefix,
                'index_col_name': 'Name',
                'mixed_compartment_list': self.mixed_compartment_list,
                'to_standard_name_dict': data_metabolite_to_standard_name_dict}}

    @staticmethod
    def project_name_generator(tissue, condition, index):
        return '{}__{}__{}'.format(tissue, condition, index)

    def add_data_sheet(self, sheet_name, current_data_dict):
        final_result_dict = self.complete_dataset
        if sheet_name == DataType.test:
            final_result_dict[DataType.test] = current_data_dict
        else:
            current_information_dict = self.sheet_name_dict[sheet_name]
            tissue_name = current_information_dict[Keyword.tissue]
            if tissue_name not in final_result_dict:
                final_result_dict[tissue_name] = {}
            for data_label, specific_data_dict in current_data_dict.items():
                raw_condition, index_str = data_label.split('_')
                condition = Keyword.condition_mapping_dict[raw_condition]
                if condition not in final_result_dict[tissue_name]:
                    final_result_dict[tissue_name][condition] = {}
                final_result_dict[tissue_name][condition][int(index_str)] = specific_data_dict

    def _complete_return_dataset(self, param_dict):
        tissue_name = param_dict[Keyword.tissue]
        condition = param_dict[Keyword.condition]
        index_num = param_dict[Keyword.index]
        if index_num == CommonKeywords.average:
            final_target_metabolite_data_dict = average_mid_data_dict(
                self.complete_dataset[tissue_name][condition], Keyword.index_average_list)
        else:
            final_target_metabolite_data_dict = self.complete_dataset[tissue_name][condition][index_num]
        final_input_metabolite_data_dict = None
        project_name = self.project_name_generator(tissue_name, condition, index_num)
        return project_name, final_target_metabolite_data_dict, final_input_metabolite_data_dict

    def _test_return_dataset(self, param_dict=None):
        final_target_metabolite_data_dict = self.complete_dataset[DataType.test][self.test_col]
        final_input_metabolite_data_dict = None
        project_name = DataType.test
        return project_name, final_target_metabolite_data_dict, final_input_metabolite_data_dict

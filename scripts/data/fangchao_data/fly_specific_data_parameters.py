from .data_metabolite_to_standard_name_dict import data_metabolite_to_standard_name_dict
from scripts.src.common.config import Direct, DataType, Keywords as CommonKeywords
from scripts.src.core.common.classes import TransformDict
from ..complete_dataset_class import CompleteDataset
from ..common_functions import average_mid_data_dict


class Keyword(object):
    labeling = 'labeling'
    tissue = 'tissue'
    age = 'age'
    condition = 'condition'
    index = 'index'

    sucrose = 'sucrose'
    glutamine = 'glutamine'

    whole_body = 'wholebody'
    sperm = 'sperm'

    adult = 'adult'
    old = 'old'

    ctrl = 'ctrl'
    mr = 'mr'
    mr_fa = 'mr_fa'

    prefix = 'prefix'

    ctrl_mr_mrfa = 'ctrl_mr_mrfa'

    age_mapping_dict = TransformDict(**{
        'Adult': adult,
        'Old': old,
    })
    condition_mapping_dict = TransformDict(**{
        'Ctrl': ctrl,
        'MR': mr,
        'MR/FA': mr_fa,
    })
    index_average_list = [1, 2, 3]


class SpecificParameters(CompleteDataset):
    def __init__(self):
        super(SpecificParameters, self).__init__()
        self.mixed_compartment_list = ('c', 'm')
        self.current_direct = "{}/fangchao_data".format(Direct.data_direct)
        self.file_path = "{}/13C_labeling_data_Fangchao.xlsx".format(self.current_direct)
        self.sheet_name_dict = {
            'sucrose_wholebody': {
                Keyword.labeling: Keyword.sucrose,
                Keyword.tissue: Keyword.whole_body,
            },
            'sucrose_sperm': {
                Keyword.labeling: Keyword.sucrose,
                Keyword.tissue: Keyword.sperm,
            },
            'glutamine_wholebody': {
                Keyword.labeling: Keyword.glutamine,
                Keyword.tissue: Keyword.whole_body,
            },
            'glutamine_sperm': {
                Keyword.labeling: Keyword.glutamine,
                Keyword.tissue: Keyword.sperm,
            }
        }
        self.test_experiment_name_prefix = "sucrose_wholebody"
        self.test_col = 'Adult_Ctrl_1'
        self._complete_data_parameter_dict_dict = {
            current_sheet_name: {
                'xlsx_file_path': self.file_path,
                'xlsx_sheet_name': current_sheet_name,
                'index_col_name': "Name",
                'mixed_compartment_list': self.mixed_compartment_list,
                'to_standard_name_dict': data_metabolite_to_standard_name_dict}
            for current_sheet_name in self.sheet_name_dict.keys()}
        self._test_data_parameter_dict_dict = {
            DataType.test: {
                'xlsx_file_path': self.file_path,
                'xlsx_sheet_name': self.test_experiment_name_prefix,
                'index_col_name': "Name",
                'mixed_compartment_list': self.mixed_compartment_list,
                'to_standard_name_dict': data_metabolite_to_standard_name_dict}}

    @staticmethod
    def project_name_generator(labeling, tissue, age, condition, index):
        return '{}__{}__{}__{}__{}'.format(labeling, tissue, age, condition, index)

    def add_data_sheet(self, sheet_name, current_data_dict):
        final_result_dict = self.complete_dataset
        if sheet_name == DataType.test:
            final_result_dict[DataType.test] = current_data_dict
        else:
            current_information_dict = self.sheet_name_dict[sheet_name]
            labeling_name = current_information_dict[Keyword.labeling]
            tissue_name = current_information_dict[Keyword.tissue]
            if labeling_name not in final_result_dict:
                final_result_dict[labeling_name] = {}
            if tissue_name not in final_result_dict[labeling_name]:
                final_result_dict[labeling_name][tissue_name] = {}
            for data_label, specific_data_dict in current_data_dict.items():
                raw_age, raw_condition, index_str = data_label.split('_')
                age = Keyword.age_mapping_dict[raw_age]
                if age not in final_result_dict[labeling_name][tissue_name]:
                    final_result_dict[labeling_name][tissue_name][age] = {}
                condition = Keyword.condition_mapping_dict[raw_condition]
                if condition not in final_result_dict[labeling_name][tissue_name][age]:
                    final_result_dict[labeling_name][tissue_name][age][condition] = {}
                final_result_dict[labeling_name][tissue_name][age][condition][int(index_str)] = specific_data_dict

    def _complete_return_dataset(self, param_dict):
        labeling_name = param_dict[Keyword.labeling]
        tissue_name = param_dict[Keyword.tissue]
        age = param_dict[Keyword.age]
        condition = param_dict[Keyword.condition]
        index_num = param_dict[Keyword.index]
        if index_num == CommonKeywords.average:
            final_target_metabolite_data_dict = average_mid_data_dict(
                self.complete_dataset[labeling_name][tissue_name][age][condition], Keyword.index_average_list)
        else:
            final_target_metabolite_data_dict = self.complete_dataset[
                labeling_name][tissue_name][age][condition][index_num]
        final_input_metabolite_data_dict = None
        project_name = self.project_name_generator(labeling_name, tissue_name, age, condition, index_num)
        return project_name, final_target_metabolite_data_dict, final_input_metabolite_data_dict

    def _test_return_dataset(self, param_dict=None):
        final_target_metabolite_data_dict = self.complete_dataset[
            DataType.test][self.test_col]
        final_input_metabolite_data_dict = None
        project_name = DataType.test
        return project_name, final_target_metabolite_data_dict, final_input_metabolite_data_dict

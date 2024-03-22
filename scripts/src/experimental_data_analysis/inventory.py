from ..common.built_in_packages import ValueEnum


class DataModelType(ValueEnum):
    fangchao_data_fruit_fly = 'fangchao_data_fruit_fly'
    fangchao_data_cultured_cell = 'fangchao_data_cultured_cell'


data_model_comment = {
    DataModelType.fangchao_data_fruit_fly:
        'Data from fruit flies treated with isotope-labeled diet',
    DataModelType.fangchao_data_cultured_cell:
        'Data from cultured cells treated with isotope-labeled media',
}


class RunningMode(ValueEnum):
    flux_analysis = 'flux_analysis'
    result_process = 'result_process'
    raw_experimental_data_plotting = 'raw_experimental_data_plotting'
    solver_output = 'solver_output'


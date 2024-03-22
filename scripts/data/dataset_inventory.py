from .config import DataType


def return_dataset_and_keyword(dataset_name):
    if dataset_name == DataType.fangchao_fly_data:
        from .fangchao_data.fly_specific_data_parameters import SpecificParameters, Keyword
    elif dataset_name == DataType.fangchao_cultured_cell_data:
        from .fangchao_data.cultured_cell_specific_data_parameters import SpecificParameters, Keyword
    else:
        raise ValueError()
    dataset_obj = SpecificParameters()
    return dataset_obj, Keyword

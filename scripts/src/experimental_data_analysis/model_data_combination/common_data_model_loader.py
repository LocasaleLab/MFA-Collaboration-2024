from ..inventory import DataModelType


def common_data_model_function_loader(model_name):
    if model_name == DataModelType.fangchao_data_fruit_fly:
        from . import fangchao_data_fruit_fly as data_model_object
    elif model_name == DataModelType.fangchao_data_cultured_cell:
        from . import fangchao_data_cultured_cell as data_model_object
    else:
        raise ValueError()
    return data_model_object

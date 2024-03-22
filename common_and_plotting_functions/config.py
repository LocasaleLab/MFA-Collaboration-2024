from .built_in_packages import copy


class Keywords(object):
    flux_raw_data = 'flux_raw_data'
    mid_raw_data = 'mid_raw_data'
    solver_descriptions = 'solver_descriptions'
    model_metabolites_reactions_standard_name = 'model_metabolites_reactions_standard_name'


class Direct(object):
    common_data_direct = 'common_data'
    common_submitted_raw_data_direct = f'{common_data_direct}/raw_data'
    figure_raw_data_direct = f'{common_data_direct}/figure_raw_data'


class FigureDataKeywords(object):
    raw_model_distance = 'raw_model_distance'
    raw_model_raw_solution = 'raw_model_raw_solution'
    mid_comparison = 'mid_comparison'
    loss_data_comparison = 'loss_data_comparison'
    best_solution = 'best_solution'
    embedding_visualization = 'embedding_visualization'
    time_data_distribution = 'time_data_distribution'
    flux_comparison = 'flux_comparison'
    raw_flux_value_dict = 'raw_flux_value_dict'
    all_fluxes_relative_error = 'all_fluxes_relative_error'


class DefaultDict(dict):
    def __init__(self, default_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_value = default_value

    def __getitem__(self, item):
        if not super().__contains__(item):
            super().__setitem__(item, copy.deepcopy(self.default_value))
        return super().__getitem__(item)



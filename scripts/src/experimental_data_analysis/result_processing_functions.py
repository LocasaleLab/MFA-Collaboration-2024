from scripts.src.common.classes import FinalResult
from scripts.src.common.plotting_functions import group_violin_box_distribution_plot, group_bar_plot, \
    scatter_plot_for_simulated_result, FigurePlotting
from scripts.src.common.config import Color, Direct, Keywords
from scripts.src.common.result_processing_functions import experimental_mid_prediction
from common_and_plotting_functions.functions import check_and_mkdir_of_direct
from common_and_plotting_functions.config import FigureDataKeywords
from ..common.config import FigureData
from ..common.result_processing_functions import loss_data_distribution_plotting, \
    reconstruct_and_filter_data_dict
from ..common.result_output_functions import output_raw_flux_data, output_predicted_mid_data

from . import config


figure_plotting: FigurePlotting = None


def initialize_figure_plotting():
    global figure_plotting
    if figure_plotting is not None:
        return
    from figures.figure_content.common.config import ParameterName
    from figures.figure_content.common.elements import Elements
    figure_plotting = FigurePlotting(ParameterName, Elements)


def result_label_generator(mfa_data):
    return mfa_data.data_name


class CurrentFinalResult(FinalResult):
    def __init__(
            self, project_output_direct, common_data_output_direct, result_name, data_model_object):
        super(CurrentFinalResult, self).__init__(
            project_output_direct, common_data_output_direct, result_name)
        self.data_model_object = data_model_object

    def final_process(self, result_process_func):
        final_mapping_dict = self.data_model_object.collect_results(self)
        result_process_func(self, final_mapping_dict)


def normal_result_process(final_result_obj, final_mapping_dict):
    data_model_object = final_result_obj.data_model_object
    final_information_dict = final_result_obj.final_information_dict
    result_name = final_result_obj.result_name
    final_loss_data_dict = final_result_obj.final_loss_data_dict
    final_solution_data_dict = final_result_obj.final_solution_data_dict
    final_predicted_data_dict = final_result_obj.final_predicted_data_dict
    final_flux_name_index_dict = final_result_obj.final_flux_name_index_dict
    final_target_experimental_mid_data_dict = final_result_obj.final_target_experimental_mid_data_dict

    subset_index_dict = loss_data_distribution_plotting(
        result_name, final_loss_data_dict,
        output_direct=final_result_obj.this_result_output_direct, loss_percentile=config.loss_percentile)
    important_flux_display(
        result_name, final_solution_data_dict, final_mapping_dict,
        data_model_object, final_flux_name_index_dict,
        final_result_obj.flux_comparison_output_direct, subset_index_dict=subset_index_dict)
    experimental_mid_prediction(
        result_name, {Keywords.optimized: final_predicted_data_dict},
        final_result_obj.final_target_experimental_mid_data_dict, final_result_obj.mid_prediction_output_direct,
        subset_index_dict=subset_index_dict)
    try:
        figure_config_dict = data_model_object.figure_config_dict
    except AttributeError:
        figure_config_dict = None
    mid_grid_plotting(
        result_name, final_predicted_data_dict, data_model_object,
        final_result_obj.mid_prediction_output_direct, figure_config_dict)
    metabolic_network_plotting(
        data_model_object, final_solution_data_dict,
        final_flux_name_index_dict, final_result_obj.metabolic_network_visualization_direct,
        subset_index_dict=subset_index_dict)
    output_raw_flux_data(
        final_result_obj.flux_result_output_xlsx_path, final_loss_data_dict,
        final_solution_data_dict, final_flux_name_index_dict,
        final_result_obj.final_information_dict, subset_index_dict=subset_index_dict, other_label_column_dict=None)
    output_predicted_mid_data(
        final_result_obj.mid_prediction_result_output_xlsx_path, final_loss_data_dict, final_predicted_data_dict,
        final_target_experimental_mid_data_dict, final_information_dict, subset_index_dict=subset_index_dict)


def experimental_mid_and_raw_data_plotting(
        complete_experimental_mid_data_obj_dict, result_information_dict, final_result_obj):
    final_result_obj.data_model_object.experimental_data_plotting(
        complete_experimental_mid_data_obj_dict, result_information_dict,
        final_result_obj.raw_and_mid_experimental_data_display_direct)


def important_flux_display(
        result_name, raw_solution_data_dict, final_mapping_dict, data_model_object, final_flux_name_index_dict,
        flux_comparison_output_direct, subset_index_dict=None):
    reconstructed_solution_data_dict, reconstructed_flux_name_index_dict = reconstruct_and_filter_data_dict(
        raw_solution_data_dict, final_flux_name_index_dict, final_mapping_dict, subset_index_dict)
    final_dict_for_comparison, final_key_name_parameter_dict, final_color_dict = \
        data_model_object.flux_comparison_parameter_generator(
            reconstructed_solution_data_dict, reconstructed_flux_name_index_dict)
    for comparison_name, data_dict_for_plotting in final_dict_for_comparison.items():
        current_labeling_data_output_folder = '{}/{}'.format(flux_comparison_output_direct, comparison_name)
        check_and_mkdir_of_direct(current_labeling_data_output_folder)
        color_dict = final_color_dict[comparison_name]
        group_violin_box_distribution_plot(
            data_dict_for_plotting, nested_color_dict=color_dict, nested_median_color_dict=color_dict,
            title_dict=None, output_direct=current_labeling_data_output_folder, ylim=None, figsize=None,
            xaxis_rotate=True, figure_type='box')
    figure_raw_data = FigureData(FigureDataKeywords.flux_comparison, result_name)
    figure_raw_data.save_data(
        final_dict_for_comparison=final_dict_for_comparison,
        final_key_name_parameter_dict=final_key_name_parameter_dict)


def metabolic_network_plotting(
        data_model_object, final_solution_data_dict, final_flux_name_index_dict, figure_output_direct,
        subset_index_dict=None):
    figure_size = (8.5, 8.5)
    (
        experimental_mid_metabolite_set, experimental_mixed_mid_metabolite_set, biomass_metabolite_set,
        input_metabolite_set, c13_labeling_metabolite_set, boundary_flux_set, infusion
    ) = data_model_object.metabolic_network_parameter_generator()
    for raw_result_label, raw_solution_data_array in final_solution_data_dict.items():
        if subset_index_dict is not None:
            subset_index = subset_index_dict[raw_result_label]
            solution_data_array = raw_solution_data_array[subset_index]
        else:
            solution_data_array = raw_solution_data_array
        current_data_array = solution_data_array.mean(axis=0)
        current_reaction_value_dict = {
            flux_name: current_data_array[flux_index]
            for flux_name, flux_index in final_flux_name_index_dict[raw_result_label].items()}

        output_file_path = f'{figure_output_direct}/metabolic_network_{raw_result_label}.pdf'
        figure_plotting.metabolic_flux_model_function(
            output_file_path, figure_size,
            input_metabolite_set, c13_labeling_metabolite_set, experimental_mid_metabolite_set,
            experimental_mixed_mid_metabolite_set,
            biomass_metabolite_set, boundary_flux_set, current_reaction_value_dict=current_reaction_value_dict,
            infusion=infusion)


def mid_grid_plotting(
        result_name, final_predicted_data_dict, data_model_object, mid_prediction_output_direct,
        figure_config_dict=None):
    figure_size = (8.5, 11)
    if figure_config_dict is None:
        figure_config_dict = {}
    for result_label in final_predicted_data_dict.keys():
        figure_plotting.mid_prediction_function(
            result_name, result_label, data_model_object.mid_name_list, mid_prediction_output_direct,
            figure_config_dict, figure_size)

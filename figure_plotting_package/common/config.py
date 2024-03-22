float_type = 'float64'
axis_for_test = False


class ParameterName(object):
    # Basic shape parameter
    name = 'name'
    shape = 'shape'
    circle = 'circle'
    rectangle = 'rectangle'
    height_to_width_ratio = 'height_to_width_ratio'
    bottom_left = 'bottom_left'
    size = 'size'
    width = 'width'
    color = 'color'
    height = 'height'
    center = 'center'
    alpha = 'alpha'
    edge_width = 'edge_width'
    edge_style = 'edge_style'
    edge_color = 'edge_color'
    face_color = 'face_color'
    fill = 'fill'
    z_order = 'z_order'
    join_style = 'join_style'
    path_step_list = 'path_step_list'
    closed = 'closed'
    cw = 'cw'
    ccw = 'ccw'
    square_bottom_left_point = 'square_bottom_left_point'
    figure_title = 'figure_title'
    figure_subtitle = 'figure_subtitle'
    figure_title_config_dict = 'figure_title_config_dict'
    subfigure = 'subfigure'
    subfigure_label = 'subfigure_label'
    background = 'background'
    transform = 'transform'
    start = 'start'
    end = 'end'
    patch = 'patch'

    # Text
    text = 'text'
    string = 'string'
    font = 'font'
    font_size = 'font_size'
    font_color = 'font_color'
    font_weight = 'font_weight'
    font_style = 'font_style'
    horizontal = 'horizontal'
    vertical = 'vertical'
    horizontal_alignment = 'horizontal_alignment'
    vertical_alignment = 'vertical_alignment'
    angle = 'angle'
    line_space = 'line_space'

    # Arrow
    tail = 'tail'
    mid = 'mid'
    head = 'head'
    radius = 'radius'
    theta_head = 'theta_head'
    theta_tail = 'theta_tail'
    head_len_width_ratio = 'head_len_width_ratio'
    head_arrow = 'head_arrow'
    tail_arrow = 'tail_arrow'
    stem_location = 'stem_location'
    terminal_location = 'terminal_location'
    arrow = 'arrow'
    dash = 'dash'
    head_width = 'head_width'
    stem_width = 'stem_width'
    gap_line_pair_list = 'gap_line_pair_list'
    dash_solid_empty_width = 'dash_solid_empty_width'
    transition_point_list = 'transition_point_list'
    branch_list = 'branch_list'
    text_box = 'text_box'
    text_box_config = 'text_box_config'
    tail_end_center = 'tail_end_center'
    theta_tail_end_center = 'theta_tail_end_center'
    arrow_head_direction = 'arrow_head_direction'
    left_tail = 'left_tail'
    right_tail = 'right_tail'

    x_label_format_dict = 'x_label_format_dict'
    y_label_format_dict = 'y_label_format_dict'
    x_tick_label_format_dict = 'x_tick_label_format_dict'
    y_tick_label_format_dict = 'y_tick_label_format_dict'
    top_x_label_format_dict = 'top_x_label_format_dict'
    right_y_label_format_dict = 'right_y_label_format_dict'
    top_x_tick_label_format_dict = 'top_x_tick_label_format_dict'
    right_y_tick_label_format_dict = 'right_y_tick_label_format_dict'
    x_tick_separator_format_dict = 'x_tick_separator_format_dict'
    y_tick_separator_format_dict = 'y_tick_separator_format_dict'
    x_tick_separator_label_format_dict = 'x_tick_separator_label_format_dict'
    y_tick_separator_label_format_dict = 'y_tick_separator_label_format_dict'

    other_obj = 'other_obj'

    # Axis
    axis = 'axis'
    visible = 'visible'
    x = 'x'
    y = 'y'
    ax_top = 'top'
    ax_bottom = 'bottom'
    ax_left = 'left'
    ax_right = 'right'
    ax_height = 'ax_height'
    ax_width = 'ax_width'
    axis_tick_label_distance = 'axis_tick_label_distance'
    axis_label_distance = 'axis_label_distance'
    axis_label_location = 'axis_label_location'
    axis_tick_length = 'axis_tick_length'
    axis_line_start_distance = 'axis_line_start_distance'
    axis_line_end_distance = 'axis_line_end_distance'
    ax_total_bottom_left = 'ax_total_bottom_left'
    ax_total_size = 'ax_total_size'
    ax_interval = 'ax_interval'
    legend_center = 'legend_center'
    legend_area_size = 'legend_ax_size'
    legend_width_ratio = 'legend_width_ratio'
    legend_color_dict = 'legend_color_dict'
    percentage = 'percentage'
    decimal_num = 'decimal_num'
    twin_x_axis = 'twin_x_axis'
    broken_y_axis = 'broken_y_axis'
    broken_point_y_lim = 'broken_point_y_lim'

    total_width = 'total_width'
    scale = 'scale'
    bottom_left_offset = 'bottom_left_offset'
    base_z_order = 'base_z_order'
    z_order_increment = 'z_order_increment'
    cap_size = 'cap_size'       # For error bar
    data_location_cap = 'data_location_cap'
    text_axis_loc_pair = 'text_axis_loc_pair'

    # Metabolic network
    network_type = 'network_type'
    normal_network = 'normal_network'
    exchange_network = 'exchange_network'
    input_metabolite_set = 'input_metabolite_set'
    c13_labeling_metabolite_set = 'c13_labeling_metabolite_set'
    mid_data_metabolite_set = 'mid_data_metabolite_set'
    mixed_mid_data_metabolite_set = 'mixed_mid_data_metabolite_set'
    biomass_metabolite_set = 'biomass_metabolite_set'
    boundary_flux_set = 'boundary_flux_set'
    display_flux_name = 'display_flux_name'
    reaction_text_dict = 'reaction_text_dict'
    reaction_text_config_dict = 'reaction_text_config_dict'
    reaction_raw_value_dict = 'reaction_raw_value_dict'
    extra_parameter_dict = 'extra_parameter_dict'
    all_data_mode = 'all_data_mode'
    all_mixed_data_mode = 'all_mixed_data_mode'
    infusion = 'infusion'
    text_label = 'text_label'
    visualize_flux_value = 'visualize_flux_value'
    transparency = 'transparency'
    absolute_value_output_value_dict = 'absolute_value_output_value_dict'
    metabolic_network_config_dict = 'metabolic_network_config_dict'
    metabolic_network_legend_config_dict = 'metabolic_network_legend_config_dict'
    metabolic_network_text_comment_config_dict = 'metabolic_network_text_comment_config_dict'
    hidden_metabolite_set = 'hidden_metabolite_set'
    hidden_reaction_set = 'hidden_reaction_set'
    metabolite_data_sensitivity_state_dict = 'metabolite_data_sensitivity_state_dict'
    reaction_data_sensitivity_state_dict = 'reaction_data_sensitivity_state_dict'
    extra_offset = 'extra_offset'
    reaction_flux_num = 'reaction_flux_num'
    total_flux_num = 'total_flux_num'
    total_mid_num = 'total_mid_num'
    mid_metabolite_num = 'mid_metabolite_num'
    small_network = 'small_network'
    zoom_in_box = 'zoom_in_box'

    # Reaction parameters
    class_name = 'class_name'
    reaction_name = 'reaction_name'
    reversible = 'reversible'
    boundary_flux = 'boundary_flux'

    # Diagram specific parameter
    mode = 'mode'
    separate = 'separate'
    normal = 'normal'
    simulated = 'simulated'
    simulated_reoptimization = 'simulated_reoptimization'
    simulated_without_reoptimization = 'simulated_without_reoptimization'
    optimization_from_average_solutions = 'optimization_from_average_solutions'
    experimental = 'experimental'
    sensitivity = 'sensitivity'
    data_sensitivity = 'data_sensitivity'
    random_optimized_comparison = 'random_optimized_comparison'
    distribution_type = 'distribution_type'
    global_optimum = 'global_optimum'
    local_optimum = 'local_optimum'
    one_dominant_global_optimum = 'one_dominant_global_optimum'
    multiple_similar_local_optima = 'multiple_similar_local_optima'
    repeats = 'repeats'
    loss = 'loss'
    distance = 'distance'
    raw_distance = 'raw_distance'
    net_distance = 'net_distance'
    optimized = 'optimized'
    with_re_optimization = 'with_re_optimization'
    unoptimized = 'unoptimized'
    raw_optimized = 'raw_optimized'
    selected = 'selected'
    averaged = 'averaged'
    small_data = 'small_data'
    medium_data = 'medium_data'
    different_simulated_distance = 'different_simulated_distance'

    # Data figure specific parameter
    figure_data = 'figure_data'
    data_name = 'data_name'
    figure_class = 'figure_class'
    figure_type = 'figure_type'
    all_flux = 'all_flux'
    raw_flux_diff_vector = 'raw_flux_diff_vector'
    net_euclidean_distance = 'net_euclidean_distance'
    flux_absolute_distance = 'flux_absolute_distance'
    flux_relative_distance = 'flux_relative_distance'
    time_data = 'time_data'
    loss_data = 'loss_data'
    low_height = 'low_height'
    solution_distance_data = 'solution_distance_data'
    comparison_name = 'comparison_name'
    mean = 'mean'
    std = 'std'
    flux_name = 'flux_name'
    marker_size = 'marker_size'
    column_width = 'column_width'
    class_width = 'class_width'
    color_dict = 'color_dict'
    selection_size = 'selection_size'
    selection_ratio = 'selection_ratio'
    optimized_size = 'optimized_size'

    figure_config_dict = 'figure_config_dict'
    subplot_name_list = 'subplot_name_list'
    subplot_name_dict = 'subplot_name_dict'
    subplot_name_text_format_dict = 'subplot_name_text_format_dict'
    result_label = 'result_label'
    result_label_layout_list = 'result_label_layout_list'
    mid_name_list = 'mid_name_list'
    flux_name_list = 'flux_name_list'
    display_flux_name_dict = 'display_flux_name_dict'
    display_group_name_dict = 'display_group_name_dict'
    cell_line_name_list = 'cell_line_name_list'
    flux_name_location_nested_list = 'flux_name_location_nested_list'
    figure_data_parameter_dict = 'figure_data_parameter_dict'
    raw_data_figure_parameter_dict = 'raw_data_figure_parameter_dict'
    all_data_figure_parameter_dict = 'all_data_figure_parameter_dict'
    loss_data_figure_parameter_dict = 'loss_data_figure_parameter_dict'
    net_distance_data_figure_parameter_dict = 'net_distance_data_figure_parameter_dict'
    compare_one_by_one = 'compare_one_by_one'
    p_value = 'p_value'

    x_lim_list = 'x_lim_list'
    common_x_label = 'common_x_label'
    x_label_list = 'x_label_list'
    x_ticks_list = 'x_ticks_list'
    x_tick_labels_list = 'x_tick_labels_list'
    y_lim_list = 'y_lim_list'
    common_y_label = 'common_y_label'
    y_label_list = 'y_label_list'
    y_ticks_list = 'y_ticks_list'
    y_tick_labels_list = 'y_tick_labels_list'
    top_x_label_list = 'top_x_label_list'
    top_x_ticks_list = 'top_x_ticks_list'
    top_x_tick_labels_list = 'top_x_tick_labels_list'
    right_y_label_list = 'right_y_label_list'
    right_y_ticks_list = 'right_y_ticks_list'
    right_y_tick_labels_list = 'right_y_tick_labels_list'

    tick_separator_dict_list = 'tick_separator_dict_list'
    x_tick_separator_locs = 'x_tick_separator_locs'
    x_tick_separator_labels = 'x_tick_separator_labels'
    x_tick_separator_label_locs = 'x_tick_separator_label_locs'
    y_tick_separator_locs = 'y_tick_separator_locs'
    y_tick_separator_labels = 'y_tick_separator_labels'
    y_tick_separator_label_locs = 'y_tick_separator_label_locs'

    body_props = 'body_props'
    min_max_props = 'min_max_props'
    median_props = 'median_props'

    data_figure_axes = 'data_figure_axes'
    hidden_data_axes = 'hidden_data_axes'
    legend = 'legend'
    name_dict = 'name_dict'
    horiz_or_vertical = 'horiz_or_vertical'
    supplementary_text_list = 'supplementary_text_list'
    supplementary_text_loc_list = 'supplementary_text_loc_list'
    supplementary_text_format_dict = 'supplementary_text_format_dict'
    p_value_y_value_list = 'p_value_y_value_list'
    p_value_cap_parameter_dict = 'p_value_cap_parameter_list'
    text_y_offset = 'text_y_offset'
    cap_y_offset = 'cap_y_offset'
    hidden_axis_boundary = 'hidden_axis_boundary'
    hidden_axis_tick = 'hidden_axis_tick'

    default_y_tick_label_list = 'default_y_tick_label_list'
    common_y_lim = 'common_y_lim'
    common_y_lim_2 = 'common_y_lim_2'
    y_abs_lim = 'y_abs_lim'
    y_tick_interval = 'y_tick_interval'
    y_tick_interval_2 = 'y_tick_interval_2'

    legend_type = 'legend_type'
    patch_legend = 'patch_legend'
    legend_config_dict = 'legend_config_dict'
    text_config_dict = 'text_config_dict'
    total_horiz_edge_ratio = 'total_horiz_edge_ratio'
    col_horiz_edge_ratio = 'col_horiz_edge_ratio'
    total_verti_edge_ratio = 'total_verti_edge_ratio'
    row_verti_edge_ratio = 'row_verti_edge_ratio'
    legend_patch_config_dict = 'legend_patch_config_dict'
    location_config_dict = 'location_config_dict'

    error_bar = 'error_bar'
    error_bar_param_dict = 'error_bar_param_dict'
    scatter_line = 'scatter_line'

    cbar = 'cbar'
    cbar_config = 'cbar_config'

    with_single_optimized_solutions = 'with_single_optimized_solutions'
    with_collected_optimized_set = 'with_collected_optimized_set'
    with_unoptimized_set = 'with_unoptimized_set'

    # Optimization from averaged solutions parameters
    with_sloppiness_diagram = 'with_sloppiness_diagram'


class Constants(object):
    computation_eps = 1e-10


import argparse


def main():
    # parser = argparse.ArgumentParser(
    #     prog='Analysis of experimental data',
    #     description='Code for analysis of multiple experimental data by Shiyu Liu.')
    from scripts.src.experimental_data_analysis.experimental_data_analysis_main import arg_setting as \
        experiments_arg_setting
    parser = experiments_arg_setting()

    args = parser.parse_args()
    try:
        current_func = args.func
    except AttributeError:
        parser.print_help()
    else:
        args.func(args)


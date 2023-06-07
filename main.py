"""
python3.exe main.py -l {pipe_file} -p {new_hosp.csv} -o {output.csv}
python3.exe main.py -l {pipe_file} -p {new_hosp.csv new_hosp_2.csv} -o {output.csv output_2.csv}
python3.exe main.py -t {train_file1.csv train_file2.csv} -p {new_hosp.csv new_hosp_2.csv} -o {output.csv output_2.csv}
"""

from src import AutoMap as utils
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Process AutoMap files.')
parser.add_argument('-t', '--train-files',
                    metavar='train_files',
                    nargs='*',
                    help='files for training the pipeline')
parser.add_argument('-p', '--predict-files',
                    metavar='predict_files',
                    nargs='*',
                    help='file to predict the labels for')
parser.add_argument('-o', '--output-file',
                    metavar='output_file',
                    nargs='*',
                    help='file to write results to')
parser.add_argument('-l', '--pipeline',
                    metavar='pipe_file',
                    nargs=1,
                    help='file containing pre-trained pipeline')


args = parser.parse_args()

if __name__ == "__main__":
    if args.train_files:
        print(f'Training on files {args.train_files}')
    if args.predict_files:
        print(f'Predicting on {args.predict_files}')
    if args.output_file:
        print(f'Outputting to {args.output_file}')
    else:
        args.output_file = ['output.csv']
    if args.pipeline:
        print(f'Loading previous Pipeline from {args.pipeline}')
        args.pipeline = args.pipeline[0]

    am_kwargs = {'train_files': args.train_files,
                 'predict_files': args.predict_files,
                 'output_files': args.output_file,
                 'pipe_file': args.pipeline}

    am = utils.AutoMap(**am_kwargs)

    for file, output_file in zip(args.predict_files, args.output_file):
        if os.path.isfile(file):
            ext = file.split('.')[-1]
            if ext == 'csv':
                df_pred = pd.read_csv(file)
            elif ext == 'xls' or ext == 'xlsx':
                df_pred = pd.read_excel(file)
            else:
                assert False, f'Format {ext} not supported; use a .csv or .xls or .xlsx file'

            output_data = am.pipe.predict(data=df_pred)
            output_data.to_csv(output_file)





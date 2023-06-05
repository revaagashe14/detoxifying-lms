#! /usr/bin/env python
import argparse
import numpy as np
from twitteraae.code.predict import *
import csv

def make_predict_files(input_path, toxic_aae_path, nontoxic_aae_path, toxic_wae_path, nontoxic_wae_path, other_path):
    toxic_aae = []
    toxic_wae = []
    nontoxic_aae = []
    nontoxic_wae = []
    other = []
    # African-American, Hispanic, Asian, White
    with open(input_path) as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i%1000 == 0:
                print("Line: ", i)
            text = line.strip().split("\t")
            load_model()
            p = predict(text[1].split())
            if p is None:
                continue
            index = max(enumerate(p), key=lambda x: x[1])[0]
            # african-american
            if index == 0:
                if int(text[0]) == 0:
                    nontoxic_aae.append(text[1])
                else:
                    toxic_aae.append(text[1])
            
            # white
            elif index == 3:
                if int(text[0]) == 0:
                    nontoxic_wae.append(text[1])
                else:
                    toxic_wae.append(text[1])
            
            # asian, hispanic
            else:
                other.append(text[1])

    pathsDict = {toxic_aae_path: toxic_aae, nontoxic_aae_path: nontoxic_aae, 
        toxic_wae_path: toxic_wae, nontoxic_wae_path: nontoxic_wae, other_path: other}

    for out_path, predictions in pathsDict.items():
        f = open(out_path, "w")
        for pred in predictions:
            f.write(pred + "\n")
        print("Wrote to file {}".format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="path to input file",
    )
    parser.add_argument(
        "--toxic_aae_path",
        type=str,
        default="",
        help="path to aae file",
    )  
    parser.add_argument(
        "--nontoxic_aae_path",
        type=str,
        default="",
        help="path to aae file",
    )  
    parser.add_argument(
        "--toxic_wae_path",
        type=str,
        default="",
        help="path to wae file",
    )  
    parser.add_argument(
        "--nontoxic_wae_path",
        type=str,
        default="",
        help="path to wae file",
    )    
    parser.add_argument(
        "--other_path",
        type=str,
        default="",
        help="path to asian/hispanic file",
    )
    args = parser.parse_args()
    make_predict_files(**vars(args))

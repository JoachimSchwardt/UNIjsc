#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:46:28 2022

@author: joachim
"""

import os


def get_newlines(filename, keywords=["\bra"], filetype=".tex", 
                 Newcommand=False, Comments=False):
    """
    Read the lines from the given filename and remove all keywords.
    
    Newcommand == bool, whether to remove 'newcommand' lines containing a keyword
    Comments == bool, whether to remove 'comment' lines (starting with %)
    """
    
    with open(filename + filetype, 'r') as f:
        lines = f.readlines()
        for kw in keywords:
            for i, line in enumerate(lines):
                if Newcommand:
                    if line.startswith("\\newcommand"):
                        if kw in line:
                            lines[i] = ""
                
                if Comments:
                    if line.startswith("%"):
                        lines[i] = ""
                    
                lines[i] = line.replace(kw, "")
                
    return lines
        
def clean_file(filename, keywords=["\bra"], filetype=".tex"):
    lines = get_newlines(filename, keywords, filetype)
    with open(filename + filetype, 'w') as f:
        f.writelines(lines)


def main():
    keywords = [r"\bra", r"\brak", r"\braket", r"\intx", r"\intr", r"\kla", 
                r"\klam", r"\klammer", r"\oper", r"\tecxt", r"\Bra", r"\Ket"]
    num2_path = "/home/joachim/Documents/UNI/Mathematik/Numerik_II/NUM2_Latex/"
    filename = num2_path + "01_GaussSeidel"
    clean_file(filename, keywords)
    return 0

if __name__ == "__main__":
    main()
    
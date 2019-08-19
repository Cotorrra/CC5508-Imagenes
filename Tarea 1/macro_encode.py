import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Macro Encoder de Texto')

parser.add_argument('--image', action="store", type=str, nargs=1, required=True,
                    help='Indica la imagen sobre la que se codificará/decodificará un texto')
parser.add_argument('--text', action="store", type=str, nargs=1, default="",
                    help='Indica el archivo que contiene el texto a codificar.')

args = parser.parse_args()
image = args.image[0]
text = args.text[0]

for i in range(1,9):
    command = "python tarea_1.py --encode --nbits "+str(i)+" --image "+image+" --text "+text
    a = subprocess.run(command)
    if a.returncode == 0 :
        ex_pos = len(image) - image[::-1].find('.') - 1
        old_filename = image[:ex_pos] + "out" + image[ex_pos:]
        new_filename = image[:ex_pos] + str(i) + image[ex_pos:]
        os.rename(old_filename, new_filename)
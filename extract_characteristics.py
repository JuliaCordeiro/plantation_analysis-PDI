from matplotlib import pyplot as plt
from turtle import color
from datetime import datetime
import cv2
import numpy as np
import os, sys

print(datetime.now())

path = "./image_first_cut"

with open('data_for_analysis.txt', 'w') as f:

    f.write( '%Trabalho da Disciplina PDI\n' )
    f.write( '\n' )

    f.write( '@relation AndreJuliaMurilo_Hist16niveisCinza\n' )
    f.write( '\n' )

    f.write( '@attribute id NUMERIC\n' )
    for x in range(16):
      f.write( f'@attribute cor{x+1} NUMERIC\n' )

    f.write( '@attribute classe {T.1, T.2, T.3, T.4, T.5, T.6, T.7, T.8}\n' )
    f.write( '\n' )

    f.write( '@DATA\n' )
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            imagename = filename[:-4]

            lead_T = imagename[5:8]
            leaf_id = imagename[11:]

            img = cv2.imread(path + "/" + filename)
            histogram = cv2.calcHist([img], [0], None, [16], [0, 16])

            histogram_value = ''

            for x in range(16):
                histogram_value = histogram_value + ', ' + f'{int(histogram[x][0])}'

            f.write( f'{leaf_id}{histogram_value}, {lead_T}\n' )


print(datetime.now())

cv2.waitKey(0)

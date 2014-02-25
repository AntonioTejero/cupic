#!/bin/bash

### USER CONFIGURATION (USER ACCESIBLE) ###

PARTICLE_TIPE=electron
INI=1
FIN=100000
INC=1

### SCRIPT CONFIGURATION (NOT USER ACCESIBLE) ###

FILENAME=bins_"$PARTICLE_TIPE"_movie 

### GENERATION OF GNUPLOT SCRIPT ###

echo set terminal jpeg size 1280,720 >> plot.gpi

echo j=0 >> plot.gpi
echo do for[i=$INI:$FIN:$INC] \{ >> plot.gpi
  
echo set output \""$FILENAME"_\".j.\".jpg\" >> plot.gpi
echo set nokey >> plot.gpi
echo set grid >> plot.gpi

#echo set xrange [0:12.7] >> plot.gpi
#echo set yrange [0:102.1] >> plot.gpi
echo set ylabel \"bin\" >> plot.gpi
echo set xlabel \"$PARTICLE_TIPE\" >> plot.gpi
echo set title \"Which bin every "$PARTICLE_TIPE" is \(t = \".i.\"\)\" >> plot.gpi
echo plot \'./bins_"$PARTICLE_TIPE"s_t_\'.i.\'.dat\' lc rgb \"blue\" >> plot.gpi

echo j=j+1 >> plot.gpi

echo \} >> plot.gpi

### EXECUTE GNUPLOT SCRIPT FOR FRAMES GENERATION ###

gnuplot plot.gpi

### REMOVE GNUPLOT SCRIPT ###

rm plot.gpi

### GENERATE MOVIE FROM FRAMES AND REMOVE FRAMES ###

avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
find . -name '*.jpg' -type f -print -delete

#!/bin/bash

FILENAME=particles_position
INI=0
FIN=80628
INC=10

echo set terminal jpeg size 1280,720 > plot.gpi

#echo set xrange[0:12.8] >> plot.gpi
#echo set yrange[0:102.2] >> plot.gpi

for ((i=$INI, j=0; i<=$FIN; i=$i+$INC, j++))
do
  echo set output \""$FILENAME"_$j.jpg\" >> plot.gpi
 
  echo set nokey >> plot.gpi
  echo set size 1,1 >> plot.gpi
  echo set origin 0,0 >> plot.gpi
  echo set multiplot >> plot.gpi

  echo set tmargin 0.5 >> plot.gpi
  echo set bmargin 0.5 >> plot.gpi
  echo set rmargin 1.0 >> plot.gpi
  echo set lmargin 5.0 >> plot.gpi

#   echo set xrange [0:12.7] >> plot.gpi
#   echo set yrange [0:102.1] >> plot.gpi
  echo set ylabel \"y\" >> plot.gpi
  echo set xlabel \"x\" >> plot.gpi
  echo set grid >> plot.gpi

  echo set size 0.45,0.85 >> plot.gpi
  echo set origin 0.03,0.07 >> plot.gpi
  echo unset key >> plot.gpi
  echo -e set title \"electrons t = $i\" >> plot.gpi
  echo -e plot \'./electrons_t_$i.dat\' lc rgb \"blue\" >> plot.gpi

  echo set size 0.45,0.85 >> plot.gpi
  echo set origin 0.51,0.07 >> plot.gpi
  echo -e set title \"ions t = $i\" >> plot.gpi
  echo -e plot \'./ions_t_$i.dat\' lc rgb \"red\" >> plot.gpi

  echo unset multiplot >> plot.gpi
done

gnuplot plot.gpi
rm plot.gpi
avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
rm *.jpg

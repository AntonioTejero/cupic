#!/bin/bash

FILENAME=particles_velocities
INI=0
FIN=76200
INC=100

echo set terminal jpeg size 1280,720 > plot.gpi

echo set format y \"\" >> plot.gpi
echo unset ylabel >> plot.gpi
echo set xlabel \"velocity\" >> plot.gpi
  
echo ebinwidth=0.1 >> plot.gpi
echo ibinwidth=0.001 >> plot.gpi
echo bin\(x,width\)=width*floor\(x/width\) >> plot.gpi

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
  
  echo set size 0.45,0.85 >> plot.gpi
  echo set origin 0.03,0.07 >> plot.gpi
#   echo set xrange[0:25] >> plot.gpi
  echo -e set title \"electrons t = $i\" >> plot.gpi
  echo -e plot \'./electrons_t_$i.dat\' u \(bin\(sqrt\(\$3**2+\$4**2\),ebinwidth\)\):\(1.0\) smooth freq with boxes lc rgb \"blue\" >> plot.gpi

  echo set origin 0.51,0.07 >> plot.gpi
#   echo set xrange[0:0.35] >> plot.gpi
  echo -e set title \"ions t = $i\" >> plot.gpi
  echo -e plot \'./ions_t_$i.dat\' u \(bin\(sqrt\(\$3**2+\$4**2\),ibinwidth\)\):\(1.0\) smooth freq with boxes lc rgb \"red\" >> plot.gpi

  echo unset multiplot >> plot.gpi
done

gnuplot plot.gpi
rm plot.gpi
avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
find . -name '*.jpg' -type f -print -delete

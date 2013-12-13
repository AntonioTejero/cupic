#!/bin/bash

FILENAME=charge
INI=9
FIN=300000
INC=10

echo set terminal jpeg size 1280,720 > plot.gpi

echo set xrange[0:128] >> plot.gpi
echo set yrange[0:1022] >> plot.gpi
echo set zrange[-5000:5000] >> plot.gpi
echo set cbrange[-5000:5000] >> plot.gpi

echo set xlabel \"i\" >> plot.gpi
echo set ylabel \"j\" >> plot.gpi
echo set zlabel \"$FILENAME\" >> plot.gpi
echo set grid >> plot.gpi

for ((i=$INI, j=0; i<=$FIN; i=$i+$INC, j++))
do
  echo -e set output \""$FILENAME"_$j.jpg\" >> plot.gpi
  echo -e splot \'./"$FILENAME"_t_$i.dat\' w pm3d >> plot.gpi
done

gnuplot plot.gpi
rm plot.gpi
avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
rm *.jpg
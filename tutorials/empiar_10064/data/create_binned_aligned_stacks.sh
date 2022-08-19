#!/bin/bash

FILES=$(ls mixedCTEM_tomo[0-9].mrc)

for ST_FULL in ${FILES}
do

  ST_BASE=${ST_FULL%.mrc}
  ST_B1=${ST_BASE}.b1.ali.mrc
  ST_B2=${ST_BASE}.b2.ali.mrc
  ST_B4=${ST_BASE}.b4.ali.mrc
  ST_B8=${ST_BASE}.b8.ali.mrc
  newstack -in ${ST_FULL} -ou ${ST_B1} -of 0.0,0.0 -x ${ST_BASE}.xf -or -ta 1,0
  newstack -in ${ST_B1} -ou ${ST_B2} -ftr 2
  newstack -in ${ST_B1} -ou ${ST_B4} -ftr 4
  newstack -in ${ST_B1} -ou ${ST_B8} -ftr 8

done


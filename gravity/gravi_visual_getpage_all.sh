#!/bin/bash

usage="
    Create PDF files containing single-page reports of all PDF files 
    of the following types separately:

    OIFITS, ASTRORED, P2VMRED, DUAL_SKY, SINGLE_SKY, RAW

    OPTIONS
    -------
    -o : Overwrite all files; If not used, skip the types 
         if gravi_series_page_all-typename exists.
"

if [[ "$1" == "-o" ]]; then
   echo "Overwrite all files"

    for typename in 'OIFITS' 'ASTRORED' 'P2VMRED' 'DUAL_SKY' 'SINGLE_SKY' 'RAW'
    do
        if [ -d gravi_series_page_all-$typename ]; then
            echo 'gravi_series_page_all-'$typename' exists; Remove it!'
            rm -rf gravi_series_page_all-$typename
        fi
    done
fi

for typename in 'OIFITS' 'ASTRORED' 'P2VMRED' 'DUAL_SKY' 'SINGLE_SKY' 'RAW'
do
    if [ -d gravi_series_page_all-$typename ]; then
        echo 'gravi_series_page_all-'$typename' exists; Skip!'
        continue
        #rm -rf gravi_series_page_all-$typename
    fi
    
    gravi_visual_getpage_new.sh all *$typename.pdf

    if [ -d 'gravi_series_page_all' ]; then
        mv gravi_series_page_all gravi_series_page_all-$typename
    else
        echo 'No '$typename' reports!'
    fi
done

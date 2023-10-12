#!/bin/bash


usage="
    $(basename "$0") pagenum report-pdf-files

    Use cpdf to create a report containing a single page (pagenum)
    extract from all PDF files report-pdf-files

    OR

    With option 'all', create a folder 'gravi_series_page_all' and 
    split the reports into single-page files for all pages.

    ex:

        $(basename "$0") 4 *OIFITS.pdf
   
        OR

        $(basename "$0") all *OIFITS.pdf

"

if [[ "$1" == "" ]]; then
   echo "$usage"
   exit
fi

if [[ "$1" == "-h" ]]; then
   echo "$usage"
   exit
fi

arr=($@)

if [[ -e "${arr[1]}" ]]; then
    totalpages_1=$(cpdf -pages ${arr[1]})
    echo 'Total pages: ' $totalpages_1
else
    echo 'cannot find any file ('${arr[1]}')!'
    exit
fi

#
# Extract page from all PDF
#

if [[ "$1" == "all" ]]; then
	echo "Split all pages"
# Extract all pages
	for (( i=1 ; i<=$totalpages_1 ; i++ )); 
	do
		echo "Extract page: " $i

		pagenum="0"
		for arg in "$@"
		do
			if [[ $pagenum == "0" ]] 
			then 
				pagenum=$i
			else
    			file="${arg%.*}"
    			ext="${arg##*.}"
    
    			if [[ $ext == "pdf" ]]
    			then 
    				echo "Extract page "$pagenum" from "$file
    				# pdftk $file".pdf" cat $pagenum output $file"-pagenumtmp"$pagenum".pdf"
    				cpdf $file".pdf" $pagenum -o $file"-pagenumtmp"$pagenum".pdf"
    			fi
    		fi
    	done
		#
		#
		# Merge them into a single PDF document
		#
		# pdftk `ls *-pagenumtmp"$pagenum".pdf` cat output "gravi_series_page"$pagenum".pdf"
		# pdftk *"-pagenumtmp"$pagenum".pdf" cat output "gravi_series_page"$pagenum".pdf"
		cpdf *"-pagenumtmp"$pagenum".pdf" -o "gravi_series_page"$pagenum".pdf"
		rm -rf *"-pagenumtmp"*.pdf
		echo "Create:"
		echo " gravi_series_page"$pagenum".pdf"
	done

	if [ -d 'gravi_series_page_all' ]
	then 
		rm -rf gravi_series_page_all
	fi
	mkdir gravi_series_page_all
	mv gravi_series_page*pdf gravi_series_page_all

	if [ "$(ls -A gravi_series_page_all)" ]; then
	    echo "Moved all files in gravi_series_page_all"
	else
	    rm -rf gravi_series_page_all
	    echo "No file created, removed gravi_series_page_all"
	fi

# Extract a single page
else
	pagenum="0"

	for arg in "$@"
	do
		if [[ $pagenum == "0" ]]
		then
			pagenum=$arg
		else
			file="${arg%.*}"
			ext="${arg##*.}"

	        echo '---'
	        echo "SG: -- $file -- end"
	        echo "SG: -- $ext  -- end"
	        echo '---'

			if [[ $ext == "pdf" ]]
			then 
				echo "Extract page "$pagenum" from "$file
				# pdftk $file".pdf" cat $pagenum output $file"-pagenumtmp"$pagenum".pdf"
				cpdf $file".pdf" $pagenum -o $file"-pagenumtmp"$pagenum".pdf"
			fi
		fi
	done
	#
	#
	# Merge them into a single PDF document
	#
	# pdftk `ls *-pagenumtmp"$pagenum".pdf` cat output "gravi_series_page"$pagenum".pdf"
	# pdftk *"-pagenumtmp"$pagenum".pdf" cat output "gravi_series_page"$pagenum".pdf"
	cpdf *"-pagenumtmp"$pagenum".pdf" -o "gravi_series_page"$pagenum".pdf"
	rm -rf *"-pagenumtmp"*.pdf
	echo "Create:"
	echo " gravi_series_page"$pagenum".pdf"
fi
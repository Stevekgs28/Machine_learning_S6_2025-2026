clean :
	rm -rf rapport.toc rapport.pdf rapport.log rapport.aux
	rm -rf code/*.png *.png

report :
	pdflatex rapport.tex

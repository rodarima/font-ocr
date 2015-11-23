2015N23

Introducción
------------
Las tipografías presentan unas formas únicas que caracterizan a cada uno de los
símbolos que contienen. Analizando las imágenes tomadas de documentos impresos,
es posible obtener información sobre la tipografía que se ha empleado al
escribir el documento, permitiendo en un futuro, reconstruir el documento
original de forma precisa.

Generación de la base de datos
------------------------------
En el directorio data/target hay una colección de imágenes de letras previamente
clasificadas. En data/font se encuentran las fuentes que se analizarán en
formato TrueType.

El script build-db.py genera una muestra de alta resolución para cada letra del 
alfabeto (sin la ñ) y para cada tipografía. Estas muestras son las que luego 
serán analizadas.

Análisis de las imágenes
------------------------
El programa analyze.py trata de obtener la fuente y el tamaño de las letras 
impresas comparando cada tipografía.


Ejecución
---------
Construir la base de datos y ejecutar en análisis:

	$ python2 build-db.py
	$ python2 analyze.py

Resultados
----------
El programa de análisis muestra unos resultados como los siguientes:

	cmb10	cmbx10	cmbx12	cmmi5	cmmi10	cmmi12	cmr5	cmr9	cmr10	cmr12	f  s  Target file
	60.4%	76.9%	81.2%	23.3%	21.3%	21.2%	61.6%	60.1%	56.1%	53.1%	+f +s data/target/cmbx12/a/uned1.png
	55.7%	73.5%	73.6%	0.0%	28.0%	25.9%	76.3%	67.1%	62.1%	58.8%	+f +s data/target/cmr5/a/a.png
	55.2%	55.4%	61.6%	0.0%	24.6%	25.0%	51.2%	85.7%	79.5%	75.4%	+f +s data/target/cmr9/a/a.png
	59.4%	53.4%	57.1%	0.0%	30.5%	30.1%	5.6%	80.0%	78.1%	74.0%	+f -s data/target/cmr10/a/big1.png
	52.2%	52.4%	58.5%	0.0%	26.9%	24.9%	0.0%	86.6%	84.8%	79.8%	+f -s data/target/cmr10/a/small.png
	66.7%	61.9%	66.9%	0.0%	29.7%	30.1%	10.4%	77.8%	73.8%	69.9%	+f -s data/target/cmr10/a/texbook.png

Las columnas "f" y "s" determinan si el reconocimiento de la fuente y el tamaño
han tenido éxito con un +f y +s respectivamente. En caso contrario con -f y -s.

En la tabla se observa como en todas las muestras se ha reconocido la fuente,
pero sólo se ha determinado correctamente el tamaño en 3 de 6.

Para más detalles de los resultados, abrir el archivo resultados.txt

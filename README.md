# Active Learning para clasificador de opinion

##### Greidy Valdes C-512
##### Ariel Bazán  C-512

###Introducción

Se propone un

### Datos

No fue posible encontrar ningún corpus en espñol con comentarios y su posterior clasificación en Positivo o Negativo. Por esto, fue necesario crear este corpus. En medio de este proceso se observó que los comentarios no solamente podían ser clasificados en Positivos y Negativos, ya que existían algunos realizando preguntas o añadiendo información. Otros, eran parte positivos y parte negativos. Por esto se decidió añadir dos nuevas opciones y clasificar los primeros como Objetivos y los segundos como Neutros.
De esta forma, utilizando los comentarios de Cubadebate acerca de las 70 tiendas en USD en el país, se construyó un conjunto de datos de 650 comentarios, de los cuales ....

Como se puede apreciar, el conjunto alcanzado fue muy pequeño y desbalanceado

### Active Learning

Se pensó que la técnica de Active Learning es la más adecuada para el problema en cuestión por varias razones. Esta estrategia suele funcionar bien para conjuntos de datos pequeños ya que el propio clasificador selecciona que datos añadir al conjunto de entrenamiento para definir mejor la superficie de decisión. Lo componen X elementos principales:
    1. S Estrategia de selección: Es la encargada de seleccionar el próximo elemento a añadir al conjunto de entrenamiento
    2. O Oráculo: Es el que sabe cuál es la verdadera clasificación de un elemento del conjunto de datos. Puede ser, para un conjunto de datos, la clasificación ya conocina de los mismos, o puede necesitar de la intervención humana.
    3. Sc Criterio de parada: Determina hasta cuando seguir añadiendo elementos al conjunto de entrenamiento
    4. L y U Etiquetados y no etiquetados: Conjuntos de elementos etiquetados y no etiquetados respectivamente en cada iteración del proceso de Active Learning
Existe otro elemento que aunque en la literatura no constituye un elemento para active learning en la gran mayoría de los casos es utilizado por la estrategia de selección. Este elemento es el clasificador. 

El clasificador usualmente además de predecir la clase a la que pertenece un elemento permite estimar la probabilidad de que realmente esta sea la correcta. Esto es utilizado por la estategia de selección para escoger, por egemplo, los casos con menor probabilidad.

(Poner pseudocodigo AL??)




